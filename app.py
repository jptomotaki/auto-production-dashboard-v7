import math
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Auto Production Planning Dashboard",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            background: rgba(248, 249, 251, 0.85);
            border: 1px solid rgba(120, 120, 120, 0.16);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .summary-box {
            border: 1px solid rgba(49, 130, 206, 0.30);
            background: rgba(49, 130, 206, 0.08);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 12px;
        }
        .assumption-box {
            border-left: 4px solid #7a7a7a;
            background: rgba(127, 127, 127, 0.07);
            border-radius: 7px;
            padding: 12px 16px;
            margin: 8px 0 14px 0;
        }
        .small-note {
            font-size: 0.90rem;
            opacity: 0.82;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# FIXED BUSINESS PERIODS AND SETTINGS
# =====================================================

HISTORICAL_MONTHS = 6
DEFAULT_FORECAST_MONTHS = 5

NUMBER_OF_SIMULATIONS = 10_000
BUDGET_SEARCH_SIMULATIONS = 1_500
BUDGET_SEARCH_STEP = 100

BASELINE_MONTHLY_BUDGET = 3_600
DEFAULT_VEHICLE_GOAL = 400

HISTORICAL_AUTO_POLICIES = 292
HISTORICAL_FIRE_POLICIES = 108
HISTORICAL_TOTAL_POLICIES = (
    HISTORICAL_AUTO_POLICIES + HISTORICAL_FIRE_POLICIES
)
AUTO_POLICY_SHARE = (
    HISTORICAL_AUTO_POLICIES / HISTORICAL_TOTAL_POLICIES
)

VEHICLES_PER_AUTO_POLICY = 1.5
MULTILINE_FIRE_ATTACHMENT_RATE = 0.45

# Rounded planning assumption requested by the agency.
STATEFARM_MONTHLY_LEADS = 70


# =====================================================
# HISTORICAL LEAD-SOURCE DATA
# =====================================================

sources = pd.DataFrame(
    {
        "Source": [
            "EverQuote",
            "Smart Financial",
            "Insurance Quotes",
            "StateFarm.com",
            "Referrals",
            "Winbacks",
        ],
        "Category": [
            "Paid",
            "Paid",
            "Paid",
            "Free",
            "Organic",
            "Organic",
        ],
        "Historical Leads": [193, 230, 37, 429, 214, 58],
        "Historical Quotes": [100, 170, 31, 355, 181, 44],
        "Closed Policies": [17, 8, 3, 30, 55, 8],
        "Cost Per Lead": [16.75, 13.00, 35.00, 0.00, 10.00, 5.00],
        "Assignment Hours": [8.7, 11.0, 11.5, 17.0, np.nan, np.nan],
        "Paid Budget Eligible": [True, True, True, False, False, False],
    }
)

sources["Quote Rate"] = (
    sources["Historical Quotes"] / sources["Historical Leads"]
)
sources["Quote-to-Close Rate"] = (
    sources["Closed Policies"] / sources["Historical Quotes"]
)
sources["Lead-to-Close Rate"] = (
    sources["Closed Policies"] / sources["Historical Leads"]
)
sources["Average Monthly Leads"] = (
    sources["Historical Leads"] / HISTORICAL_MONTHS
)
sources["Historical Source Cost"] = (
    sources["Historical Leads"] * sources["Cost Per Lead"]
)
sources["Historical Cost Per Close"] = np.where(
    sources["Closed Policies"] > 0,
    sources["Historical Source Cost"] / sources["Closed Policies"],
    np.nan,
)
sources["Expected Auto Closes Per Dollar"] = np.where(
    sources["Paid Budget Eligible"] & (sources["Cost Per Lead"] > 0),
    (
        sources["Lead-to-Close Rate"]
        * AUTO_POLICY_SHARE
        / sources["Cost Per Lead"]
    ),
    np.nan,
)


# =====================================================
# HISTORICAL AGENCY BASELINE
# =====================================================

TRACKED_HISTORICAL_POLICIES = int(sources["Closed Policies"].sum())
ESTIMATED_TRACKED_AUTO_POLICIES = (
    TRACKED_HISTORICAL_POLICIES * AUTO_POLICY_SHARE
)
OTHER_AGENCY_HISTORICAL_AUTO_POLICIES = (
    HISTORICAL_AUTO_POLICIES - ESTIMATED_TRACKED_AUTO_POLICIES
)

HISTORICAL_VEHICLE_ITEMS = (
    HISTORICAL_AUTO_POLICIES * VEHICLES_PER_AUTO_POLICY
)
HISTORICAL_MONTHLY_VEHICLE_PACE = (
    HISTORICAL_VEHICLE_ITEMS / HISTORICAL_MONTHS
)
# The selected forecast-period pace is calculated after the user
# chooses the number of months in the sidebar.

PAID_PROVIDER_RECORDED_MONTHLY_COST = (
    sources.loc[
        sources["Paid Budget Eligible"],
        "Historical Source Cost",
    ].sum()
    / HISTORICAL_MONTHS
)
UNMAPPED_MONTHLY_BUDGET = max(
    BASELINE_MONTHLY_BUDGET - PAID_PROVIDER_RECORDED_MONTHLY_COST,
    0,
)


# =====================================================
# VALIDATION
# =====================================================


def validate_historical_data() -> None:
    if (sources["Historical Leads"] < 0).any():
        raise ValueError("Historical leads cannot be negative.")
    if (sources["Historical Quotes"] > sources["Historical Leads"]).any():
        raise ValueError("Historical quotes cannot exceed leads.")
    if (sources["Closed Policies"] > sources["Historical Quotes"]).any():
        raise ValueError("Closed policies cannot exceed quotes.")
    if (sources["Cost Per Lead"] < 0).any():
        raise ValueError("Cost per lead cannot be negative.")
    if OTHER_AGENCY_HISTORICAL_AUTO_POLICIES < 0:
        raise ValueError(
            "Tracked-source production cannot exceed agency-wide auto production."
        )


validate_historical_data()


# =====================================================
# INCREMENTAL PAID-BUDGET ALLOCATION
# =====================================================


def calculate_incremental_allocation(
    selected_monthly_budget: float,
    apply_capacity_limits: bool,
    capacity_multiplier: float,
) -> tuple[pd.DataFrame, float]:
    """
    Preserve historical paid-source volume at the $3,600 baseline.

    Only money above $3,600 is allocated to incremental leads. The optional
    capacity multiplier limits each paid provider's total monthly lead volume
    to a multiple of its own historical monthly volume.
    """

    paid = sources[sources["Paid Budget Eligible"]].copy()
    extra_budget = max(selected_monthly_budget - BASELINE_MONTHLY_BUDGET, 0.0)

    paid["Efficiency Weight"] = paid["Expected Auto Closes Per Dollar"]
    paid["Historical Monthly Leads"] = paid["Average Monthly Leads"]

    if apply_capacity_limits:
        paid["Maximum Monthly Leads"] = (
            paid["Historical Monthly Leads"] * capacity_multiplier
        )
        paid["Incremental Lead Capacity"] = np.maximum(
            paid["Maximum Monthly Leads"] - paid["Historical Monthly Leads"],
            0.0,
        )
        paid["Incremental Dollar Capacity"] = (
            paid["Incremental Lead Capacity"] * paid["Cost Per Lead"]
        )
    else:
        paid["Maximum Monthly Leads"] = np.inf
        paid["Incremental Lead Capacity"] = np.inf
        paid["Incremental Dollar Capacity"] = np.inf

    allocations = np.zeros(len(paid), dtype=float)
    remaining_budget = float(extra_budget)
    remaining_capacity = paid["Incremental Dollar Capacity"].to_numpy(dtype=float).copy()
    efficiency = paid["Efficiency Weight"].to_numpy(dtype=float)

    # Reallocate any dollars left over when a provider reaches its capacity.
    for _ in range(20):
        active = (remaining_capacity > 1e-9) & (efficiency > 0)
        if remaining_budget <= 1e-9 or not np.any(active):
            break

        active_weights = efficiency[active]
        active_weights = active_weights / active_weights.sum()
        proposed = remaining_budget * active_weights
        spend = np.minimum(proposed, remaining_capacity[active])

        allocations[active] += spend
        remaining_capacity[active] -= spend
        amount_allocated = float(spend.sum())
        remaining_budget -= amount_allocated

        if amount_allocated <= 1e-9:
            break

    # Convert the continuous dollar allocation into whole lead purchases.
    # A provider cannot receive a fractional lead order.
    costs = paid["Cost Per Lead"].to_numpy(dtype=float)

    if apply_capacity_limits:
        whole_lead_capacity = np.floor(
            paid["Incremental Lead Capacity"].to_numpy(dtype=float)
        ).astype(int)
    else:
        whole_lead_capacity = np.full(
            len(paid),
            np.iinfo(np.int32).max,
            dtype=int,
        )

    incremental_leads = np.floor(
        np.divide(
            allocations,
            costs,
            out=np.zeros_like(allocations),
            where=costs > 0,
        )
        + 1e-9
    ).astype(int)
    incremental_leads = np.minimum(
        incremental_leads,
        whole_lead_capacity,
    )

    actual_spend = incremental_leads * costs
    remaining_budget = max(
        extra_budget - float(actual_spend.sum()),
        0.0,
    )

    # Use any affordable remainder on the most efficient available source.
    # Because the initial dollar allocation has already been rounded down,
    # this loop normally adds only a few leads.
    priority_order = np.argsort(-efficiency)
    for provider_index in priority_order:
        provider_cost = costs[provider_index]
        if provider_cost <= 0:
            continue

        remaining_provider_capacity = (
            whole_lead_capacity[provider_index]
            - incremental_leads[provider_index]
        )
        if remaining_provider_capacity <= 0:
            continue

        affordable_leads = int(
            math.floor(
                (remaining_budget + 1e-9)
                / provider_cost
            )
        )
        leads_to_add = min(
            affordable_leads,
            remaining_provider_capacity,
        )

        if leads_to_add > 0:
            incremental_leads[provider_index] += leads_to_add
            added_spend = leads_to_add * provider_cost
            actual_spend[provider_index] += added_spend
            remaining_budget = max(
                remaining_budget - added_spend,
                0.0,
            )

    paid["Incremental Monthly Budget"] = actual_spend
    paid["Incremental Budget Share"] = np.where(
        extra_budget > 0,
        paid["Incremental Monthly Budget"] / extra_budget,
        0.0,
    )
    paid["Expected Incremental Monthly Leads"] = (
        incremental_leads.astype(int)
    )
    paid["Expected Total Monthly Leads"] = (
        paid["Historical Monthly Leads"]
        + paid["Expected Incremental Monthly Leads"]
    )
    paid["Expected Auto Policies Per $1,000"] = (
        paid["Expected Auto Closes Per Dollar"] * 1_000
    )

    return (
        paid[
            [
                "Source",
                "Historical Monthly Leads",
                "Maximum Monthly Leads",
                "Incremental Budget Share",
                "Incremental Monthly Budget",
                "Expected Incremental Monthly Leads",
                "Expected Total Monthly Leads",
                "Expected Auto Policies Per $1,000",
            ]
        ],
        max(remaining_budget, 0.0),
    )


# =====================================================
# SOURCE PLAN
# =====================================================


def build_source_plan(
    selected_monthly_budget: float,
    forecast_months: int,
    apply_capacity_limits: bool,
    capacity_multiplier: float,
) -> tuple[pd.DataFrame, float]:
    """
    Build monthly lead assumptions for the selected forecast period.

    At $3,600, paid sources remain at their historical monthly lead volume.
    Below $3,600, paid lead volume is reduced proportionally.
    Above $3,600, incremental dollars create incremental paid leads.
    """

    allocation, unallocated_budget = calculate_incremental_allocation(
        selected_monthly_budget,
        apply_capacity_limits,
        capacity_multiplier,
    )
    allocation_lookup = allocation.set_index("Source").to_dict(orient="index")

    below_baseline_scale = min(
        selected_monthly_budget / BASELINE_MONTHLY_BUDGET,
        1.0,
    )

    rows = []

    for _, row in sources.iterrows():
        source_name = row["Source"]
        historical_monthly_leads = float(row["Average Monthly Leads"])
        expected_incremental_monthly_leads = 0.0
        incremental_monthly_budget = 0.0
        incremental_budget_share = 0.0

        if row["Paid Budget Eligible"]:
            if selected_monthly_budget < BASELINE_MONTHLY_BUDGET:
                expected_monthly_leads = (
                    historical_monthly_leads * below_baseline_scale
                )
                treatment = "Historical paid volume scaled down"
            else:
                allocation_row = allocation_lookup[source_name]
                expected_incremental_monthly_leads = float(
                    allocation_row["Expected Incremental Monthly Leads"]
                )
                incremental_monthly_budget = float(
                    allocation_row["Incremental Monthly Budget"]
                )
                incremental_budget_share = float(
                    allocation_row["Incremental Budget Share"]
                )
                expected_monthly_leads = (
                    historical_monthly_leads
                    + expected_incremental_monthly_leads
                )
                treatment = (
                    "Historical paid volume maintained"
                    if selected_monthly_budget == BASELINE_MONTHLY_BUDGET
                    else "Historical volume plus incremental leads"
                )

        elif source_name == "StateFarm.com":
            expected_monthly_leads = float(STATEFARM_MONTHLY_LEADS)
            treatment = "70-lead monthly planning assumption"

        else:
            expected_monthly_leads = historical_monthly_leads
            treatment = "Historical monthly volume maintained"

        rows.append(
            {
                "Source": source_name,
                "Category": row["Category"],
                "Historical Monthly Leads": historical_monthly_leads,
                "Expected Incremental Monthly Leads": (
                    expected_incremental_monthly_leads
                ),
                "Expected Monthly Leads": expected_monthly_leads,
                "Expected Forecast-Period Leads": (
                    expected_monthly_leads * forecast_months
                ),
                "Incremental Budget Share": incremental_budget_share,
                "Incremental Monthly Budget": incremental_monthly_budget,
                "Budget Treatment": treatment,
            }
        )

    return pd.DataFrame(rows), unallocated_budget


# =====================================================
# MONTE CARLO SIMULATION
# =====================================================


@st.cache_data(show_spinner=False)
def run_simulation(
    selected_monthly_budget: float,
    forecast_months: int,
    conversion_multiplier: float,
    apply_capacity_limits: bool,
    capacity_multiplier: float,
    simulations: int = NUMBER_OF_SIMULATIONS,
    seed: int = 42,
) -> dict:
    """Simulate monthly and total agency-wide auto production."""

    if selected_monthly_budget < 0:
        raise ValueError("Monthly budget cannot be negative.")
    if forecast_months < 1:
        raise ValueError("Forecast months must be at least 1.")
    if conversion_multiplier <= 0:
        raise ValueError("Conversion multiplier must be positive.")
    if capacity_multiplier < 1:
        raise ValueError("Capacity multiplier must be at least 1.0.")

    rng = np.random.default_rng(seed)
    source_plan, unallocated_budget = build_source_plan(
        selected_monthly_budget,
        forecast_months,
        apply_capacity_limits,
        capacity_multiplier,
    )

    # One auto-share draw per simulation, used across all selected forecast months.
    auto_share_draws = rng.beta(
        HISTORICAL_AUTO_POLICIES,
        HISTORICAL_FIRE_POLICIES,
        size=simulations,
    )

    # Auto production outside the six tracked sources.
    expected_other_auto_per_month = (
        OTHER_AGENCY_HISTORICAL_AUTO_POLICIES / HISTORICAL_MONTHS
    )
    other_auto_monthly = rng.poisson(
        lam=expected_other_auto_per_month,
        size=(simulations, forecast_months),
    )
    other_extra_vehicles_monthly = rng.poisson(
        lam=(VEHICLES_PER_AUTO_POLICY - 1) * other_auto_monthly
    )
    other_vehicles_monthly = (
        other_auto_monthly + other_extra_vehicles_monthly
    )
    other_multiline_monthly = rng.binomial(
        n=other_auto_monthly,
        p=MULTILINE_FIRE_ATTACHMENT_RATE,
    )

    total_auto_monthly = other_auto_monthly.copy()
    total_vehicles_monthly = other_vehicles_monthly.copy()
    total_multiline_monthly = other_multiline_monthly.copy()

    tracked_closes_monthly = np.zeros(
        (simulations, forecast_months), dtype=int
    )
    tracked_auto_monthly = np.zeros(
        (simulations, forecast_months), dtype=int
    )
    tracked_vehicles_monthly = np.zeros(
        (simulations, forecast_months), dtype=int
    )
    tracked_multiline_monthly = np.zeros(
        (simulations, forecast_months), dtype=int
    )

    source_results = []

    for _, source_row in sources.iterrows():
        source_name = source_row["Source"]
        plan_row = source_plan[source_plan["Source"] == source_name].iloc[0]
        expected_monthly_leads = float(plan_row["Expected Monthly Leads"])

        simulated_leads_monthly = rng.poisson(
            lam=max(expected_monthly_leads, 0.0),
            size=(simulations, forecast_months),
        )

        # Draw a plausible rate once per simulation, then apply it across months.
        quote_rates = rng.beta(
            float(source_row["Historical Quotes"]),
            float(
                source_row["Historical Leads"]
                - source_row["Historical Quotes"]
            ),
            size=simulations,
        )
        close_rates = rng.beta(
            float(source_row["Closed Policies"]),
            float(
                source_row["Historical Quotes"]
                - source_row["Closed Policies"]
            ),
            size=simulations,
        )
        adjusted_close_rates = np.clip(
            close_rates * conversion_multiplier,
            0.0,
            1.0,
        )

        simulated_quotes_monthly = rng.binomial(
            n=simulated_leads_monthly,
            p=quote_rates[:, None],
        )
        simulated_closes_monthly = rng.binomial(
            n=simulated_quotes_monthly,
            p=adjusted_close_rates[:, None],
        )
        simulated_auto_monthly = rng.binomial(
            n=simulated_closes_monthly,
            p=auto_share_draws[:, None],
        )
        simulated_extra_vehicles_monthly = rng.poisson(
            lam=(VEHICLES_PER_AUTO_POLICY - 1) * simulated_auto_monthly
        )
        simulated_vehicles_monthly = (
            simulated_auto_monthly + simulated_extra_vehicles_monthly
        )
        simulated_multiline_monthly = rng.binomial(
            n=simulated_auto_monthly,
            p=MULTILINE_FIRE_ATTACHMENT_RATE,
        )

        tracked_closes_monthly += simulated_closes_monthly
        tracked_auto_monthly += simulated_auto_monthly
        tracked_vehicles_monthly += simulated_vehicles_monthly
        tracked_multiline_monthly += simulated_multiline_monthly

        total_auto_monthly += simulated_auto_monthly
        total_vehicles_monthly += simulated_vehicles_monthly
        total_multiline_monthly += simulated_multiline_monthly

        source_vehicle_totals = simulated_vehicles_monthly.sum(axis=1)
        source_auto_totals = simulated_auto_monthly.sum(axis=1)
        source_close_totals = simulated_closes_monthly.sum(axis=1)
        source_quote_totals = simulated_quotes_monthly.sum(axis=1)
        source_lead_totals = simulated_leads_monthly.sum(axis=1)
        source_multiline_totals = simulated_multiline_monthly.sum(axis=1)

        source_results.append(
            {
                "Source": source_name,
                "Category": source_row["Category"],
                "Expected Leads": source_lead_totals.mean(),
                "Expected Quotes": source_quote_totals.mean(),
                "Expected Closed Policies": source_close_totals.mean(),
                "Expected Auto Policies": source_auto_totals.mean(),
                "Expected Vehicle Items": source_vehicle_totals.mean(),
                "Expected Multiline Fire Attachments": (
                    source_multiline_totals.mean()
                ),
                "10th Percentile Vehicle Items": np.percentile(
                    source_vehicle_totals, 10
                ),
                "Median Vehicle Items": np.percentile(
                    source_vehicle_totals, 50
                ),
                "90th Percentile Vehicle Items": np.percentile(
                    source_vehicle_totals, 90
                ),
                "Incremental Forecast-Period Cost": (
                    float(plan_row["Incremental Monthly Budget"])
                    * forecast_months
                ),
            }
        )

    return {
        "source_plan": source_plan,
        "source_summary": pd.DataFrame(source_results),
        "unallocated_incremental_budget": unallocated_budget,
        "vehicle_items_monthly": total_vehicles_monthly,
        "auto_policies_monthly": total_auto_monthly,
        "multiline_fire_monthly": total_multiline_monthly,
        "vehicle_items": total_vehicles_monthly.sum(axis=1),
        "auto_policies": total_auto_monthly.sum(axis=1),
        "multiline_fire_attachments": total_multiline_monthly.sum(axis=1),
        "tracked_closed_policies": tracked_closes_monthly.sum(axis=1),
        "tracked_auto_policies": tracked_auto_monthly.sum(axis=1),
        "tracked_vehicle_items": tracked_vehicles_monthly.sum(axis=1),
        "tracked_multiline_fire": tracked_multiline_monthly.sum(axis=1),
        "other_agency_auto_policies": other_auto_monthly.sum(axis=1),
        "other_agency_vehicle_items": other_vehicles_monthly.sum(axis=1),
        "other_agency_multiline_fire": other_multiline_monthly.sum(axis=1),
    }


# =====================================================
# BUDGET SEARCH
# =====================================================


@st.cache_data(show_spinner=False)
def build_budget_curve(
    vehicle_goal: int,
    forecast_months: int,
    selected_probability: float,
    maximum_budget: int,
    conversion_multiplier: float,
    apply_capacity_limits: bool,
    capacity_multiplier: float,
) -> tuple[pd.DataFrame, dict]:
    """Test budgets and return expected, 50%, and selected-confidence thresholds."""

    rows = []

    for tested_budget in range(
        0,
        maximum_budget + BUDGET_SEARCH_STEP,
        BUDGET_SEARCH_STEP,
    ):
        result = run_simulation(
            selected_monthly_budget=float(tested_budget),
            forecast_months=forecast_months,
            conversion_multiplier=conversion_multiplier,
            apply_capacity_limits=apply_capacity_limits,
            capacity_multiplier=capacity_multiplier,
            simulations=BUDGET_SEARCH_SIMULATIONS,
            seed=2_026,
        )
        vehicles = result["vehicle_items"]
        rows.append(
            {
                "Monthly Budget": tested_budget,
                "Expected Vehicle Items": float(vehicles.mean()),
                "Vehicle Goal Probability": float(
                    np.mean(vehicles >= vehicle_goal)
                ),
            }
        )

    curve = pd.DataFrame(rows)

    def first_budget(mask: pd.Series):
        matches = curve.loc[mask, "Monthly Budget"]
        return None if matches.empty else int(matches.iloc[0])

    thresholds = {
        "expected_goal_budget": first_budget(
            curve["Expected Vehicle Items"] >= vehicle_goal
        ),
        "fifty_percent_budget": first_budget(
            curve["Vehicle Goal Probability"] >= 0.50
        ),
        "selected_probability_budget": first_budget(
            curve["Vehicle Goal Probability"] >= selected_probability
        ),
    }

    return curve, thresholds


# =====================================================
# HELPERS
# =====================================================


def curve_row_for_budget(curve: pd.DataFrame, budget) -> pd.Series | None:
    if budget is None:
        return None
    matches = curve[curve["Monthly Budget"] == budget]
    return None if matches.empty else matches.iloc[0]



def budget_text(value) -> str:
    return "Not reached" if value is None else f"${value:,.0f}/month"





# =====================================================
# HEADER AND SIDEBAR
# =====================================================

st.title("Auto Production Planning Dashboard")
st.caption(
    "Historical inputs use January–June performance. Choose a forecast period, "
    "vehicle goal, monthly paid-lead budget, and conversion scenario."
)

st.sidebar.header("Scenario")
st.sidebar.caption(
    "Change the settings, then click Run Forecast."
)

with st.sidebar.form(
    "scenario_settings_form",
    clear_on_submit=False,
):
    forecast_months = int(
        st.number_input(
            "Forecast period in months",
            min_value=1,
            value=DEFAULT_FORECAST_MONTHS,
            step=1,
        )
    )

    vehicle_goal = st.number_input(
        "Vehicle-item goal",
        min_value=0,
        max_value=100_000,
        value=DEFAULT_VEHICLE_GOAL,
        step=10,
    )

    selected_monthly_budget = st.number_input(
        "Monthly paid-lead budget",
        min_value=0,
        max_value=100_000,
        value=BASELINE_MONTHLY_BUDGET,
        step=100,
        format="%d",
    )

    conversion_label = st.selectbox(
        "Conversion scenario",
        options=[
            "Conservative: 10% below historical",
            "Historical conversion",
            "Improved: 10% above historical",
        ],
        index=1,
        help=(
            "This changes quote-to-close performance only. Historical conversion "
            "uses the observed January–June rates."
        ),
    )

    st.form_submit_button(
        "Run Forecast",
        use_container_width=True,
        type="primary",
    )

# Keep the interface simple while preserving an 80% confidence reference.
desired_probability_percent = 80
desired_probability = 0.80
maximum_budget_to_test = 12_000
apply_capacity_limits = False
capacity_multiplier = 2.0

forecast_period_label = (
    "1-month forecast"
    if forecast_months == 1
    else f"{forecast_months}-month forecast"
)
selected_month_numbers = list(range(1, forecast_months + 1))
historical_pace_forecast = (
    HISTORICAL_MONTHLY_VEHICLE_PACE * forecast_months
)

conversion_multiplier_lookup = {
    "Conservative: 10% below historical": 0.90,
    "Historical conversion": 1.00,
    "Improved: 10% above historical": 1.10,
}
conversion_multiplier = conversion_multiplier_lookup[conversion_label]


# =====================================================
# RUN SELECTED AND BASELINE SCENARIOS
# =====================================================

with st.spinner("Running production scenarios..."):
    selected_simulation = run_simulation(
        selected_monthly_budget=float(selected_monthly_budget),
        forecast_months=int(forecast_months),
        conversion_multiplier=conversion_multiplier,
        apply_capacity_limits=apply_capacity_limits,
        capacity_multiplier=float(capacity_multiplier),
        simulations=NUMBER_OF_SIMULATIONS,
        seed=42,
    )

    baseline_simulation = run_simulation(
        selected_monthly_budget=float(BASELINE_MONTHLY_BUDGET),
        forecast_months=int(forecast_months),
        conversion_multiplier=conversion_multiplier,
        apply_capacity_limits=apply_capacity_limits,
        capacity_multiplier=float(capacity_multiplier),
        simulations=NUMBER_OF_SIMULATIONS,
        seed=42,
    )

with st.spinner("Comparing budget levels..."):
    probability_curve, budget_thresholds = build_budget_curve(
        vehicle_goal=int(vehicle_goal),
        forecast_months=int(forecast_months),
        selected_probability=desired_probability,
        maximum_budget=int(maximum_budget_to_test),
        conversion_multiplier=conversion_multiplier,
        apply_capacity_limits=apply_capacity_limits,
        capacity_multiplier=float(capacity_multiplier),
    )

vehicle_results = selected_simulation["vehicle_items"]
auto_results = selected_simulation["auto_policies"]
monthly_vehicle_results = selected_simulation["vehicle_items_monthly"]
source_plan = selected_simulation["source_plan"]
source_summary = selected_simulation["source_summary"]

expected_vehicle_items = float(vehicle_results.mean())
expected_auto_policies = float(auto_results.mean())
vehicle_goal_probability = float(np.mean(vehicle_results >= vehicle_goal))

baseline_expected_vehicles = float(
    baseline_simulation["vehicle_items"].mean()
)
incremental_vehicle_effect = (
    expected_vehicle_items - baseline_expected_vehicles
)
remaining_gap = vehicle_goal - expected_vehicle_items

expected_goal_budget = budget_thresholds["expected_goal_budget"]
fifty_percent_budget = budget_thresholds["fifty_percent_budget"]
selected_probability_budget = budget_thresholds[
    "selected_probability_budget"
]


# =====================================================
# FORECAST CONTEXT
# =====================================================

summary_direction = (
    "above" if expected_vehicle_items >= vehicle_goal else "below"
)
summary_difference = abs(expected_vehicle_items - vehicle_goal)

st.caption(
    f"{forecast_period_label}  •  Historical pace: "
    f"{historical_pace_forecast:,.0f} vehicles  •  "
    f"Selected budget: ${selected_monthly_budget:,.0f}/month  •  "
    f"Baseline budget: ${BASELINE_MONTHLY_BUDGET:,.0f}/month"
)


# =====================================================
# TOP KPI ROW
# =====================================================

k1, k2, k3 = st.columns(3)
k1.metric(
    "Expected Vehicle Items",
    f"{expected_vehicle_items:,.1f}",
    delta=f"{expected_vehicle_items - vehicle_goal:+,.1f} vs goal",
)
k2.metric(
    "Probability of Reaching Goal",
    f"{vehicle_goal_probability:.1%}",
)
k3.metric(
    "Expected Auto Policies",
    f"{expected_auto_policies:,.1f}",
)


# =====================================================
# INSIGHT SUMMARY
# =====================================================

st.subheader("Insight Summary")

if selected_probability_budget is None:
    recommendation_text = (
        "The model did not reach the fixed 80% confidence reference within the "
        "tested budget range."
    )
elif selected_probability_budget > BASELINE_MONTHLY_BUDGET:
    recommendation_text = (
        f"The lowest tested budget reaching the fixed 80% confidence reference "
        f"is approximately ${selected_probability_budget:,.0f} per month."
    )
else:
    recommendation_text = (
        f"The established ${BASELINE_MONTHLY_BUDGET:,.0f} monthly budget reaches "
        "the 80% confidence reference in the model."
    )

st.markdown(
    f"""
- **Forecast:** {expected_vehicle_items:,.0f} vehicle items over {forecast_months} month{'s' if forecast_months != 1 else ''}.
- **Goal comparison:** {summary_difference:,.0f} {summary_direction} the {vehicle_goal:,}-vehicle goal.
- **Chance of reaching the goal:** {vehicle_goal_probability:.0%} of the 10,000 simulated outcomes reached or exceeded the goal.
- **Budget guidance:** {recommendation_text}
"""
)


# =====================================================
# PRELIMINARY ACTION
# =====================================================

st.subheader("Preliminary Action")

if expected_goal_budget is None:
    st.warning(
        "The model did not find a monthly budget where expected vehicle "
        f"production reaches {vehicle_goal:,} within the tested range. "
        "Increase the maximum budget tested or review the operating assumptions."
    )
else:
    expected_goal_match = probability_curve.loc[
        probability_curve["Monthly Budget"] == expected_goal_budget
    ]

    if expected_goal_match.empty:
        expected_at_action_budget = float("nan")
        probability_at_action_budget = float("nan")
    else:
        expected_at_action_budget = float(
            expected_goal_match.iloc[0]["Expected Vehicle Items"]
        )
        probability_at_action_budget = float(
            expected_goal_match.iloc[0]["Vehicle Goal Probability"]
        )

    action_budget_change = (
        expected_goal_budget - BASELINE_MONTHLY_BUDGET
    )
    action_period_change = (
        action_budget_change * forecast_months
    )

    if expected_goal_budget > BASELINE_MONTHLY_BUDGET:
        action_heading = (
            f"Use approximately ${expected_goal_budget:,.0f} per month "
            "as the preliminary planning budget."
        )
        action_detail = (
            f"This is ${action_budget_change:,.0f} more per month than the "
            f"established ${BASELINE_MONTHLY_BUDGET:,.0f} budget, or "
            f"${action_period_change:,.0f} of additional spending over the "
            f"{forecast_months}-month forecast."
        )
    elif expected_goal_budget < BASELINE_MONTHLY_BUDGET:
        action_heading = (
            f"The model reaches the goal in expectation near "
            f"${expected_goal_budget:,.0f} per month."
        )
        action_detail = (
            f"Because the established budget is ${BASELINE_MONTHLY_BUDGET:,.0f}, "
            "maintaining the current budget provides more cushion than the "
            "minimum expected-value threshold."
        )
    else:
        action_heading = (
            f"Maintain the established ${BASELINE_MONTHLY_BUDGET:,.0f} "
            "monthly budget as the preliminary action."
        )
        action_detail = (
            "The current budget is the lowest tested level where expected "
            "production reaches the selected vehicle goal."
        )

    st.markdown(
        f"""
<div class="summary-box">
    <strong>{action_heading}</strong><br><br>
    {action_detail}<br><br>
    At this budget, the model projects approximately
    <strong>{expected_at_action_budget:,.0f} vehicle items</strong> and a
    <strong>{probability_at_action_budget:.0%}</strong> probability of reaching
    the goal.
</div>
""",
        unsafe_allow_html=True,
    )

    if expected_goal_budget > BASELINE_MONTHLY_BUDGET:
        st.markdown(
            "**Recommended allocation of the additional monthly budget**"
        )

        action_allocation, action_unspent = (
            calculate_incremental_allocation(
                float(expected_goal_budget),
                apply_capacity_limits,
                float(capacity_multiplier),
            )
        )
        action_allocation = action_allocation.copy()
        action_allocation["Incremental Budget Share"] = (
            action_allocation["Incremental Budget Share"]
            * 100
        ).round(1)
        action_allocation["Incremental Monthly Budget"] = (
            action_allocation["Incremental Monthly Budget"]
            .round(2)
        )
        action_allocation["Expected Incremental Monthly Leads"] = (
            action_allocation["Expected Incremental Monthly Leads"]
            .astype(int)
        )
        action_allocation["Additional Leads Over Forecast"] = (
            action_allocation["Expected Incremental Monthly Leads"]
            * forecast_months
        ).astype(int)
        action_allocation["Expected Total Monthly Leads"] = (
            action_allocation["Expected Total Monthly Leads"]
            .round(1)
        )

        action_allocation_display = action_allocation[
            [
                "Source",
                "Incremental Budget Share",
                "Incremental Monthly Budget",
                "Expected Incremental Monthly Leads",
                "Additional Leads Over Forecast",
                "Expected Total Monthly Leads",
            ]
        ].rename(
            columns={
                "Incremental Budget Share": (
                    "Share of Extra Budget (%)"
                ),
                "Incremental Monthly Budget": (
                    "Additional Monthly Allocation ($)"
                ),
                "Expected Incremental Monthly Leads": (
                    "Additional Leads per Month"
                ),
                "Expected Total Monthly Leads": (
                    "Total Monthly Leads After Increase"
                ),
            }
        )

        st.dataframe(
            action_allocation_display,
            use_container_width=True,
            hide_index=True,
        )

        if action_unspent > 0.01:
            st.caption(
                f"${action_unspent:,.2f} per month remains unassigned "
                "because leads must be purchased as whole units"
                + (
                    " and the selected provider-capacity limits apply."
                    if apply_capacity_limits
                    else "."
                )
            )
    else:
        st.info(
            "No additional paid-lead allocation is needed at the "
            "preliminary expected-value budget."
        )

    st.caption(
        "This recommendation targets the average modeled outcome. The fixed 80% "
        "confidence budget shown below includes a larger safety margin."
    )


# =====================================================
# PATH TO GOAL
# =====================================================

st.subheader("Path to Goal")

p1, p2, p3, p4 = st.columns(4)
p1.metric(
    "Baseline Forecast at $3,600",
    f"{baseline_expected_vehicles:,.1f}",
)
p2.metric(
    "Effect of Selected Budget",
    f"{incremental_vehicle_effect:+,.1f}",
)
p3.metric(
    "Selected Forecast",
    f"{expected_vehicle_items:,.1f}",
)
p4.metric(
    "Remaining Gap",
    f"{max(remaining_gap, 0):,.1f}",
    delta=(
        "Goal reached in expectation"
        if remaining_gap <= 0
        else "Still needed"
    ),
)

progress_value = 1.0 if vehicle_goal <= 0 else min(
    expected_vehicle_items / vehicle_goal,
    1.0,
)
st.progress(progress_value)
st.caption(
    f"Historical pace is about {HISTORICAL_MONTHLY_VEHICLE_PACE:.1f} vehicle items "
    f"per month. The {forecast_months}-month goal requires "
    f"{vehicle_goal / forecast_months:.1f} per month."
)

# =====================================================
# BUDGET SCENARIO COMPARISON
# =====================================================

st.subheader("Budget Scenarios")

baseline_curve_row = curve_row_for_budget(
    probability_curve,
    BASELINE_MONTHLY_BUDGET,
)
expected_curve_row = curve_row_for_budget(
    probability_curve,
    expected_goal_budget,
)
eighty_curve_row = curve_row_for_budget(
    probability_curve,
    selected_probability_budget,
)

s1, s2, s3 = st.columns(3)

with s1:
    st.metric("Established Budget", f"${BASELINE_MONTHLY_BUDGET:,.0f}/month")
    if baseline_curve_row is not None:
        st.caption(
            f"Expected: {baseline_curve_row['Expected Vehicle Items']:.0f} vehicles  •  "
            f"Goal probability: {baseline_curve_row['Vehicle Goal Probability']:.0%}"
        )

with s2:
    st.metric("Average Forecast Reaches Goal", budget_text(expected_goal_budget))
    if expected_curve_row is not None:
        st.caption(
            f"Expected: {expected_curve_row['Expected Vehicle Items']:.0f} vehicles  •  "
            f"Goal probability: {expected_curve_row['Vehicle Goal Probability']:.0%}"
        )

with s3:
    st.metric("80% Goal Probability", budget_text(selected_probability_budget))
    if eighty_curve_row is not None:
        st.caption(
            f"Expected: {eighty_curve_row['Expected Vehicle Items']:.0f} vehicles  •  "
            f"Goal probability: {eighty_curve_row['Vehicle Goal Probability']:.0%}"
        )

st.caption(
    "The average-result budget is the first tested budget where mean production "
    "reaches the goal. The 80% budget is higher because it must cover many "
    "weaker-than-average outcomes, not only the average outcome."
)


# =====================================================
# MONTE CARLO OUTCOME DISTRIBUTION
# =====================================================

st.subheader("All 10,000 Monte Carlo Outcomes")

# A regular frequency bar chart: one bar for each exact vehicle total.
# The height of the bar is the number of simulations that produced that total.
outcome_frequency = (
    pd.Series(vehicle_results.astype(int), name="Vehicle Items")
    .value_counts()
    .sort_index()
    .rename_axis("Vehicle Items")
    .reset_index(name="Number of Simulations")
)
outcome_frequency["Share of Outcomes"] = (
    outcome_frequency["Number of Simulations"] / len(vehicle_results)
)

outcome_bars = (
    alt.Chart(outcome_frequency)
    .mark_bar(color="#2563EB", opacity=0.88, size=5)
    .encode(
        x=alt.X(
            "Vehicle Items:Q",
            title=f"Total vehicle items over {forecast_months} months",
            axis=alt.Axis(format="d", tickCount=12),
            scale=alt.Scale(zero=False),
        ),
        y=alt.Y(
            "Number of Simulations:Q",
            title="Number of simulations",
        ),
        tooltip=[
            alt.Tooltip("Vehicle Items:Q", title="Exact vehicle total", format=".0f"),
            alt.Tooltip(
                "Number of Simulations:Q",
                title="Simulations with this result",
                format=",.0f",
            ),
            alt.Tooltip(
                "Share of Outcomes:Q",
                title="Share of all outcomes",
                format=".1%",
            ),
        ],
    )
)

average_rule = (
    alt.Chart(pd.DataFrame({"Average": [expected_vehicle_items]}))
    .mark_rule(color="#111827", strokeWidth=2, strokeDash=[6, 4])
    .encode(x="Average:Q")
)

goal_rule = (
    alt.Chart(pd.DataFrame({"Goal": [float(vehicle_goal)]}))
    .mark_rule(color="#DC2626", strokeWidth=3)
    .encode(x="Goal:Q")
)

st.altair_chart(
    (outcome_bars + average_rule + goal_rule).properties(height=360),
    use_container_width=True,
)

percentile_10 = float(np.percentile(vehicle_results, 10))
median_outcome = float(np.percentile(vehicle_results, 50))
percentile_90 = float(np.percentile(vehicle_results, 90))

st.caption(
    f"Every one of the {len(vehicle_results):,} simulations is included. Each blue "
    "bar represents one exact vehicle total, and the bar height shows how many "
    "simulations produced that result. The dashed black line is the average "
    f"({expected_vehicle_items:,.1f}); the red line is the goal ({vehicle_goal:,.0f}). "
    f"The middle 80% of outcomes fall between about {percentile_10:,.0f} and "
    f"{percentile_90:,.0f} vehicles, with a median of {median_outcome:,.0f}."
)


# =====================================================
# CONTRIBUTION BREAKDOWN
# =====================================================

st.subheader("Where the Forecast Comes From")

contribution_table = pd.DataFrame(
    {
        "Production Component": [
            "Six tracked lead sources",
            "Other agency auto production",
            "Agency-wide total",
        ],
        "Expected Auto Policies": [
            selected_simulation["tracked_auto_policies"].mean(),
            selected_simulation["other_agency_auto_policies"].mean(),
            auto_results.mean(),
        ],
        "Expected Vehicle Items": [
            selected_simulation["tracked_vehicle_items"].mean(),
            selected_simulation["other_agency_vehicle_items"].mean(),
            vehicle_results.mean(),
        ],
    }
)
contribution_number_columns = [
    "Expected Auto Policies",
    "Expected Vehicle Items",
]
contribution_table[contribution_number_columns] = contribution_table[
    contribution_number_columns
].round(1)
st.dataframe(contribution_table, use_container_width=True, hide_index=True)

st.caption(
    "The 121 tracked closed policies include both auto and fire. The model first "
    "estimates the auto portion using the agency-wide 73% auto share; it does not "
    "subtract all 121 directly from the 292 auto policies."
)


# =====================================================
# INCREMENTAL ALLOCATION
# =====================================================

st.subheader("Incremental Paid-Lead Allocation")

extra_selected_budget = max(
    selected_monthly_budget - BASELINE_MONTHLY_BUDGET,
    0,
)

if selected_monthly_budget > BASELINE_MONTHLY_BUDGET:
    st.write(
        f"The first **${BASELINE_MONTHLY_BUDGET:,.0f} per month** maintains the "
        f"historical baseline. The additional **${extra_selected_budget:,.0f} per month** "
        "is allocated below."
    )

    allocation_display, allocation_unspent = calculate_incremental_allocation(
        float(selected_monthly_budget),
        apply_capacity_limits,
        float(capacity_multiplier),
    )
    allocation_display = allocation_display.copy()
    allocation_display["Incremental Budget Share"] = (
        allocation_display["Incremental Budget Share"] * 100
    ).round(1)

    for column in [
        "Historical Monthly Leads",
        "Maximum Monthly Leads",
        "Expected Total Monthly Leads",
        "Expected Auto Policies Per $1,000",
    ]:
        allocation_display[column] = allocation_display[column].replace(
            [np.inf, -np.inf], np.nan
        ).round(1)

    allocation_display["Incremental Monthly Budget"] = (
        allocation_display["Incremental Monthly Budget"]
        .round(2)
    )
    allocation_display["Expected Incremental Monthly Leads"] = (
        allocation_display["Expected Incremental Monthly Leads"]
        .astype(int)
    )
    allocation_display["Additional Leads Over Forecast"] = (
        allocation_display["Expected Incremental Monthly Leads"]
        * forecast_months
    ).astype(int)

    allocation_display = allocation_display.rename(
        columns={
            "Incremental Budget Share": "Allocated Extra Budget (%)",
            "Incremental Monthly Budget": "Incremental Monthly Allocation ($)",
            "Expected Incremental Monthly Leads": "Additional Leads per Month",
        }
    )
    st.dataframe(allocation_display, use_container_width=True, hide_index=True)

    if allocation_unspent > 1:
        st.warning(
            f"Approximately ${allocation_unspent:,.0f} of the incremental monthly "
            "budget cannot be assigned under the selected provider-capacity assumption."
        )

elif selected_monthly_budget == BASELINE_MONTHLY_BUDGET:
    st.info(
        "The selected budget equals the established $3,600 baseline, so there is "
        "no incremental budget to allocate."
    )
else:
    scale = selected_monthly_budget / BASELINE_MONTHLY_BUDGET
    st.warning(
        f"The selected budget is below baseline. Paid-source lead volume is modeled "
        f"at approximately {scale:.1%} of its historical level."
    )


# =====================================================
# MULTILINE SECTION
# =====================================================




# =====================================================
# BUDGET SENSITIVITY
# =====================================================

st.subheader("Budget Sensitivity")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    expected_chart = (
        alt.Chart(probability_curve)
        .mark_line(strokeWidth=3, color="#2563EB")
        .encode(
            x=alt.X(
                "Monthly Budget:Q",
                title="Monthly paid-lead budget ($)",
                axis=alt.Axis(format="$,.0f"),
            ),
            y=alt.Y(
                "Expected Vehicle Items:Q",
                title="Expected vehicle items",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("Monthly Budget:Q", title="Monthly budget", format="$,.0f"),
                alt.Tooltip("Expected Vehicle Items:Q", title="Expected vehicles", format=".1f"),
            ],
        )
        .properties(title="Expected production by budget", height=320)
    )
    goal_rule = (
        alt.Chart(pd.DataFrame({"Goal": [vehicle_goal]}))
        .mark_rule(color="#6B7280", strokeDash=[7, 5])
        .encode(y="Goal:Q")
    )
    st.altair_chart(expected_chart + goal_rule, use_container_width=True)

with chart_col2:
    probability_display = probability_curve.copy()
    probability_display["Goal Probability (%)"] = (
        probability_display["Vehicle Goal Probability"] * 100
    )
    probability_chart = (
        alt.Chart(probability_display)
        .mark_line(strokeWidth=3, color="#0F766E")
        .encode(
            x=alt.X(
                "Monthly Budget:Q",
                title="Monthly paid-lead budget ($)",
                axis=alt.Axis(format="$,.0f"),
            ),
            y=alt.Y(
                "Goal Probability (%):Q",
                title="Probability of reaching goal (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("Monthly Budget:Q", title="Monthly budget", format="$,.0f"),
                alt.Tooltip("Goal Probability (%):Q", title="Goal probability", format=".1f"),
            ],
        )
        .properties(title="Goal probability by budget (80% reference)", height=320)
    )
    probability_rule = (
        alt.Chart(
            pd.DataFrame(
                {"Selected Probability": [80]}
            )
        )
        .mark_rule(color="#6B7280", strokeDash=[7, 5])
        .encode(y="Selected Probability:Q")
    )
    st.altair_chart(
        probability_chart + probability_rule,
        use_container_width=True,
    )



# =====================================================
# DATA USED IN THE MODEL
# =====================================================

st.subheader("Data Used in the Model")

baseline_data = pd.DataFrame(
    {
        "Historical Input": [
            "Auto policies, January–June",
            "Fire policies, January–June",
            "Total policies, January–June",
            "Vehicle items per auto policy",
            "Estimated vehicle items, January–June",
            "Historical vehicle pace",
            "Established monthly paid-lead budget",
        ],
        "Value": [
            f"{HISTORICAL_AUTO_POLICIES:,}",
            f"{HISTORICAL_FIRE_POLICIES:,}",
            f"{HISTORICAL_TOTAL_POLICIES:,}",
            f"{VEHICLES_PER_AUTO_POLICY:.1f}",
            f"{HISTORICAL_VEHICLE_ITEMS:,.0f}",
            f"{HISTORICAL_MONTHLY_VEHICLE_PACE:.1f} vehicles per month",
            f"${BASELINE_MONTHLY_BUDGET:,.0f} per month",
        ],
    }
)
st.dataframe(baseline_data, use_container_width=True, hide_index=True)

source_data_display = sources[
    [
        "Source",
        "Historical Leads",
        "Average Monthly Leads",
        "Historical Quotes",
        "Closed Policies",
        "Quote Rate",
        "Quote-to-Close Rate",
        "Lead-to-Close Rate",
        "Cost Per Lead",
    ]
].copy()
source_data_display.insert(
    1,
    "Source Type",
    [
        "Adjustable paid",
        "Adjustable paid",
        "Adjustable paid",
        "Fixed",
        "Natural",
        "Natural",
    ],
)
source_data_display["Estimated Monthly Source Cost"] = (
    sources["Historical Source Cost"] / HISTORICAL_MONTHS
)
source_data_display["Budget Treatment"] = [
    "Additional budget can be allocated",
    "Additional budget can be allocated",
    "Additional budget can be allocated",
    "Held at 70 leads per month",
    "Held at historical pace",
    "Held at historical pace",
]

for column in ["Quote Rate", "Quote-to-Close Rate", "Lead-to-Close Rate"]:
    source_data_display[column] = (source_data_display[column] * 100).round(2)
source_data_display["Average Monthly Leads"] = source_data_display[
    "Average Monthly Leads"
].round(2)
source_data_display["Cost Per Lead"] = source_data_display[
    "Cost Per Lead"
].round(2)
source_data_display["Estimated Monthly Source Cost"] = source_data_display[
    "Estimated Monthly Source Cost"
].round(2)
source_data_display = source_data_display.rename(
    columns={
        "Historical Leads": "6-Month Leads",
        "Average Monthly Leads": "Leads per Month",
        "Historical Quotes": "Quotes",
        "Quote Rate": "Quote Rate (%)",
        "Quote-to-Close Rate": "Quote-to-Close Rate (%)",
        "Lead-to-Close Rate": "Lead-to-Close Rate (%)",
        "Cost Per Lead": "Cost per Lead ($)",
        "Estimated Monthly Source Cost": "Estimated Monthly Source Cost ($)",
    }
)
st.dataframe(source_data_display, use_container_width=True, hide_index=True)

st.info(
    "Data check: Insurance Quotes is modeled with 37 historical leads. That is the "
    "value consistent with 6.17 leads per month and a 31/37 = 83.78% quote rate. "
    "Using 47 leads would conflict with both of those figures."
)

st.markdown(
    f"""
**How the untracked auto production is calculated**

The six tracked sources produced **{TRACKED_HISTORICAL_POLICIES} total closed policies**, but those records do not identify which were auto and which were fire. Therefore, the model cannot subtract 121 directly from the 292 auto policies.

1. Estimated tracked auto policies: {TRACKED_HISTORICAL_POLICIES} × {AUTO_POLICY_SHARE:.0%} = **{ESTIMATED_TRACKED_AUTO_POLICIES:,.1f}**.
2. Estimated other agency auto policies: {HISTORICAL_AUTO_POLICIES} − {ESTIMATED_TRACKED_AUTO_POLICIES:,.1f} = **{OTHER_AGENCY_HISTORICAL_AUTO_POLICIES:,.1f}** over six months.
3. This other production is held near its historical monthly pace in the forecast.
"""
)


# =====================================================
# SOURCE DETAILS
# =====================================================

with st.expander("Source-level forecast and historical performance"):
    st.write("**Projected lead plan**")
    source_plan_display = source_plan.copy()
    for column in [
        "Historical Monthly Leads",
        "Expected Incremental Monthly Leads",
        "Expected Monthly Leads",
        "Expected Forecast-Period Leads",
        "Incremental Monthly Budget",
    ]:
        source_plan_display[column] = source_plan_display[column].round(1)
    source_plan_display["Incremental Budget Share"] = (
        source_plan_display["Incremental Budget Share"] * 100
    ).round(1)
    source_plan_display = source_plan_display.rename(
        columns={
            "Incremental Budget Share": "Incremental Budget Share (%)",
            "Incremental Monthly Budget": "Incremental Monthly Budget ($)",
        }
    )
    st.dataframe(source_plan_display, use_container_width=True, hide_index=True)

    st.write("**Tracked-source production forecast**")
    source_forecast_display = source_summary[
        [
            "Source",
            "Category",
            "Expected Leads",
            "Expected Quotes",
            "Expected Closed Policies",
            "Expected Auto Policies",
            "Expected Vehicle Items",
            "Incremental Forecast-Period Cost",
        ]
    ].copy()
    number_columns = [
        "Expected Leads",
        "Expected Quotes",
        "Expected Closed Policies",
        "Expected Auto Policies",
        "Expected Vehicle Items",
        "Incremental Forecast-Period Cost",
    ]
    source_forecast_display[number_columns] = source_forecast_display[
        number_columns
    ].round(1)
    st.dataframe(
        source_forecast_display,
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Historical lead-source performance**")
    historical_display = sources[
        [
            "Source",
            "Category",
            "Historical Leads",
            "Historical Quotes",
            "Closed Policies",
            "Cost Per Lead",
            "Quote Rate",
            "Quote-to-Close Rate",
            "Lead-to-Close Rate",
        ]
    ].copy()
    for rate_column in [
        "Quote Rate",
        "Quote-to-Close Rate",
        "Lead-to-Close Rate",
    ]:
        historical_display[rate_column] = (
            historical_display[rate_column] * 100
        ).round(1)
    historical_display = historical_display.rename(
        columns={
            "Cost Per Lead": "Cost Per Lead ($)",
            "Quote Rate": "Quote Rate (%)",
            "Quote-to-Close Rate": "Quote-to-Close Rate (%)",
            "Lead-to-Close Rate": "Lead-to-Close Rate (%)",
        }
    )
    st.dataframe(historical_display, use_container_width=True, hide_index=True)


# =====================================================
# ASSUMPTIONS AND LIMITATIONS
# =====================================================

with st.expander("Model assumptions and limitations", expanded=True):
    st.markdown(
        f"""
        <div class="assumption-box">
            <strong>What the model is doing:</strong> It uses six months of agency
            performance to create 10,000 possible future outcomes. The results are
            planning estimates, not promises.
        </div>
        """,
        unsafe_allow_html=True,
    )

    simple_assumptions = pd.DataFrame(
        {
            "Model Input": [
                "Historical period",
                "Historical vehicle pace",
                "Vehicle items per auto policy",
                "Estimated auto share of closed policies",
                "StateFarm monthly planning volume",
                "Paid sources that can receive extra budget",
                "Confidence reference shown in the app",
            ],
            "Value": [
                "January–June (6 months)",
                f"{HISTORICAL_MONTHLY_VEHICLE_PACE:.1f} vehicles per month",
                f"{VEHICLES_PER_AUTO_POLICY:.1f}",
                f"{AUTO_POLICY_SHARE:.0%}",
                f"{STATEFARM_MONTHLY_LEADS} leads per month",
                "EverQuote, Smart Financial, and Insurance Quotes",
                "80%",
            ],
        }
    )
    st.dataframe(simple_assumptions, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
### Limitations in plain language

**Only six months of history are available.**  
The model assumes January–June is a reasonable guide to the future. Seasonality, staffing changes, market conditions, or changes in customer demand could make later months different.

**The source records do not identify auto versus fire.**  
The 121 tracked closed policies include both policy types. The model estimates that {AUTO_POLICY_SHARE:.0%} are auto because {HISTORICAL_AUTO_POLICIES} of the agency's {HISTORICAL_TOTAL_POLICIES} total policies were auto.

**Not all agency production is tied to the six tracked sources.**  
The six tracked sources produced 121 total policies, while the agency produced {HISTORICAL_TOTAL_POLICIES} total policies. The remaining production is represented as “other agency production” and is assumed to continue near its historical pace.

**The full $3,600 baseline budget is not mapped to provider records.**  
The three adjustable paid sources explain about **${PAID_PROVIDER_RECORDED_MONTHLY_COST:,.0f} per month** from their recorded lead counts and costs. About **${UNMAPPED_MONTHLY_BUDGET:,.0f} per month** is not assigned to those three records. For that reason, the model treats $3,600 as the established baseline and uses cost per lead only for spending above it.

**Only three sources grow when the budget grows.**  
Additional spending purchases leads only from EverQuote, Smart Financial, and Insurance Quotes. StateFarm, referrals, winbacks, and other agency production remain at their fixed or historical pace.

**The base model assumes additional paid leads are available.**  
No confirmed provider maximums were supplied. In reality, a provider may have volume limits, price changes, or lower lead quality when more leads are purchased.

**Historical conversion may not continue.**  
The same provider can perform differently in future months. More purchased leads may not convert at the same rate as the original leads.

**A probability is not a guarantee.**  
An 80% result still means about 20% of simulated outcomes miss the goal. The simulation is meant to compare risk, not eliminate it.
"""
    )

    st.warning(
        "Use the dashboard as a planning aid alongside vendor capacity, staffing, "
        "lead quality, and management judgment."
    )
