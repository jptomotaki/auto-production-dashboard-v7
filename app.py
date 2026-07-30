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
    "Historical inputs use January–June performance. Choose any forecast length "
    "in months; the model scales the historical pace and source assumptions to "
    "that period."
)

st.sidebar.header("Scenario")
st.sidebar.caption(
    "Change as many settings as needed, then click Run Forecast once."
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
            help=(
                "Enter any whole number of months. The model runs only after "
                "you click Run Forecast."
            ),
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

    desired_probability_percent = st.slider(
        "Desired probability of reaching the goal",
        min_value=50,
        max_value=95,
        value=80,
        step=5,
    )

    conversion_label = st.selectbox(
        "Conversion scenario",
        options=[
            "Conservative: 10% below historical",
            "Historical conversion",
            "Improved: 10% above historical",
        ],
        index=1,
    )

    with st.expander("Advanced planning assumptions"):
        maximum_budget_to_test = st.number_input(
            "Maximum monthly budget to test",
            min_value=5_000,
            max_value=100_000,
            value=12_000,
            step=1_000,
        )
        apply_capacity_limits = st.checkbox(
            "Apply paid-provider lead capacity limits",
            value=False,
            help=(
                "Limits each paid provider's total monthly leads to a multiple "
                "of its historical monthly volume. This is a planning assumption, "
                "not a confirmed contractual limit."
            ),
        )
        capacity_multiplier = st.slider(
            "Maximum paid leads versus historical volume",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.25,
            help=(
                "This value is used only when paid-provider capacity limits "
                "are enabled."
            ),
        )

    st.form_submit_button(
        "Run Forecast",
        use_container_width=True,
        type="primary",
    )

forecast_period_label = (
    "1-month forecast"
    if forecast_months == 1
    else f"{forecast_months}-month forecast"
)
selected_month_numbers = list(range(1, forecast_months + 1))
historical_pace_forecast = (
    HISTORICAL_MONTHLY_VEHICLE_PACE * forecast_months
)
desired_probability = desired_probability_percent / 100

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
multiline_results = selected_simulation["multiline_fire_attachments"]
monthly_vehicle_results = selected_simulation["vehicle_items_monthly"]
source_plan = selected_simulation["source_plan"]
source_summary = selected_simulation["source_summary"]

expected_vehicle_items = float(vehicle_results.mean())
expected_auto_policies = float(auto_results.mean())
expected_multiline_fire = float(multiline_results.mean())
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

k1, k2, k3, k4 = st.columns(4)
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
k4.metric(
    "Expected Multiline Attachments",
    f"{expected_multiline_fire:,.1f}",
    help="Estimated as 45% of auto policies. It does not set the budget.",
)


# =====================================================
# INSIGHT SUMMARY
# =====================================================

st.subheader("Insight Summary")

if selected_probability_budget is None:
    recommendation_text = (
        f"The selected {desired_probability_percent}% confidence level was not "
        f"reached within the tested budget range."
    )
elif selected_probability_budget > BASELINE_MONTHLY_BUDGET:
    recommendation_text = (
        f"The lowest tested budget reaching {desired_probability_percent}% "
        f"confidence is approximately ${selected_probability_budget:,.0f} per month."
    )
else:
    recommendation_text = (
        f"The established ${BASELINE_MONTHLY_BUDGET:,.0f} monthly budget reaches "
        f"the selected confidence level in the model."
    )

st.markdown(
    f"""
- **Forecast:** {expected_vehicle_items:,.0f} vehicle items over {forecast_months} month{'s' if forecast_months != 1 else ''}.
- **Goal comparison:** {summary_difference:,.0f} {summary_direction} the {vehicle_goal:,}-vehicle goal, with a {vehicle_goal_probability:.0%} modeled probability of reaching it.
- **Budget guidance:** {recommendation_text}
- **Multiline estimate:** {expected_multiline_fire:,.0f} expected fire attachments using the {MULTILINE_FIRE_ATTACHMENT_RATE:.0%} attachment assumption; this does not drive the budget recommendation.
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
        "This recommendation targets the average modeled outcome, not the "
        "selected confidence level. Use the probability-based budget scenario "
        "when a larger safety margin is required."
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
fifty_curve_row = curve_row_for_budget(
    probability_curve,
    fifty_percent_budget,
)
selected_curve_row = curve_row_for_budget(
    probability_curve,
    selected_probability_budget,
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Established Budget", f"${BASELINE_MONTHLY_BUDGET:,.0f}/month")
    if baseline_curve_row is not None:
        st.caption(
            f"Expected: {baseline_curve_row['Expected Vehicle Items']:.0f} vehicles  •  "
            f"Goal probability: {baseline_curve_row['Vehicle Goal Probability']:.0%}"
        )

with s2:
    st.metric("Expected Result Reaches Goal", budget_text(expected_goal_budget))
    if expected_curve_row is not None:
        st.caption(
            f"Expected: {expected_curve_row['Expected Vehicle Items']:.0f} vehicles  •  "
            f"Goal probability: {expected_curve_row['Vehicle Goal Probability']:.0%}"
        )

with s3:
    st.metric("50% Goal Probability", budget_text(fifty_percent_budget))
    if fifty_curve_row is not None:
        st.caption(
            f"Expected: {fifty_curve_row['Expected Vehicle Items']:.0f} vehicles  •  "
            f"Goal probability: {fifty_curve_row['Vehicle Goal Probability']:.0%}"
        )

with s4:
    st.metric(
        f"{desired_probability_percent}% Goal Probability",
        budget_text(selected_probability_budget),
    )
    if selected_curve_row is not None:
        st.caption(
            f"Expected: {selected_curve_row['Expected Vehicle Items']:.0f} vehicles  •  "
            f"Goal probability: {selected_curve_row['Vehicle Goal Probability']:.0%}"
        )

st.caption(
    f"The expected-result budget targets an average of {vehicle_goal:,} vehicle items. "
    "The probability budgets add progressively more protection against "
    "weaker-than-average outcomes."
)


# =====================================================
# MONTE CARLO OUTCOME DISTRIBUTION
# =====================================================

st.subheader("Monte Carlo Outcome Distribution")

# Group the 10,000 simulated total-vehicle outcomes into readable ranges.
# This is a histogram: each bar shows how many simulations landed within
# that vehicle-item range.
histogram_counts, histogram_edges = np.histogram(
    vehicle_results,
    bins=36,
)

outcome_distribution = pd.DataFrame(
    {
        "Range Start": histogram_edges[:-1],
        "Range End": histogram_edges[1:],
        "Simulated Outcomes": histogram_counts,
    }
)
outcome_distribution["Outcome Share"] = (
    outcome_distribution["Simulated Outcomes"] / len(vehicle_results)
)
outcome_distribution["Range Midpoint"] = (
    outcome_distribution["Range Start"]
    + outcome_distribution["Range End"]
) / 2

outcome_bars = (
    alt.Chart(outcome_distribution)
    .mark_bar(
        opacity=0.88,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3,
    )
    .encode(
        x=alt.X(
            "Range Start:Q",
            bin="binned",
            title=(
                f"Total vehicle items across the {forecast_months}-month forecast"
            ),
        ),
        x2=alt.X2("Range End:Q"),
        y=alt.Y(
            "Simulated Outcomes:Q",
            title="Number of simulated outcomes",
        ),
        color=alt.condition(
            f"datum['Range Midpoint'] >= {float(vehicle_goal)}",
            alt.value("#16A34A"),
            alt.value("#2563EB"),
        ),
        tooltip=[
            alt.Tooltip("Range Start:Q", title="Range begins", format=".0f"),
            alt.Tooltip("Range End:Q", title="Range ends", format=".0f"),
            alt.Tooltip(
                "Simulated Outcomes:Q",
                title="Simulations",
                format=",.0f",
            ),
            alt.Tooltip(
                "Outcome Share:Q",
                title="Share of outcomes",
                format=".1%",
            ),
        ],
    )
)

expected_rule = (
    alt.Chart(
        pd.DataFrame(
            {"Expected Vehicle Items": [expected_vehicle_items]}
        )
    )
    .mark_rule(
        color="#111827",
        strokeWidth=2.5,
        strokeDash=[7, 5],
    )
    .encode(x=alt.X("Expected Vehicle Items:Q"))
)

expected_label = (
    alt.Chart(
        pd.DataFrame(
            {
                "Expected Vehicle Items": [expected_vehicle_items],
                "Label": [f"Average: {expected_vehicle_items:,.1f}"],
            }
        )
    )
    .mark_text(
        align="left",
        baseline="top",
        dx=6,
        dy=7,
        color="#111827",
        fontWeight="bold",
    )
    .encode(
        x=alt.X("Expected Vehicle Items:Q"),
        y=alt.value(0),
        text="Label:N",
    )
)

goal_rule = (
    alt.Chart(pd.DataFrame({"Vehicle Goal": [float(vehicle_goal)]}))
    .mark_rule(color="#DC2626", strokeWidth=3)
    .encode(x=alt.X("Vehicle Goal:Q"))
)

goal_label = (
    alt.Chart(
        pd.DataFrame(
            {
                "Vehicle Goal": [float(vehicle_goal)],
                "Label": [f"Goal: {vehicle_goal:,.0f}"],
            }
        )
    )
    .mark_text(
        align="left",
        baseline="top",
        dx=6,
        dy=26,
        color="#DC2626",
        fontWeight="bold",
    )
    .encode(
        x=alt.X("Vehicle Goal:Q"),
        y=alt.value(0),
        text="Label:N",
    )
)

outcome_chart = (
    outcome_bars
    + expected_rule
    + expected_label
    + goal_rule
    + goal_label
).properties(height=360)

st.altair_chart(outcome_chart, use_container_width=True)

percentile_10 = float(np.percentile(vehicle_results, 10))
median_outcome = float(np.percentile(vehicle_results, 50))
percentile_90 = float(np.percentile(vehicle_results, 90))

st.caption(
    f"Each bar groups similar results from all {len(vehicle_results):,} Monte Carlo "
    f"simulations. Blue bars fall below the {vehicle_goal:,.0f}-vehicle goal; "
    "green bars meet or exceed it. The dashed line marks the average forecast, "
    f"and the red line marks the goal. The middle 80% of outcomes range from "
    f"approximately {percentile_10:,.0f} to {percentile_90:,.0f} vehicles, with "
    f"a median of {median_outcome:,.0f}."
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
        "Expected Multiline Attachments": [
            selected_simulation["tracked_multiline_fire"].mean(),
            selected_simulation["other_agency_multiline_fire"].mean(),
            multiline_results.mean(),
        ],
    }
)
contribution_number_columns = [
    "Expected Auto Policies",
    "Expected Vehicle Items",
    "Expected Multiline Attachments",
]
contribution_table[contribution_number_columns] = contribution_table[
    contribution_number_columns
].round(1)
st.dataframe(contribution_table, use_container_width=True, hide_index=True)


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

st.subheader("Multiline Opportunity")

m1, m2, m3 = st.columns(3)
m1.metric("Expected Auto Policies", f"{expected_auto_policies:,.1f}")
m2.metric(
    "Multiline Attachment Assumption",
    f"{MULTILINE_FIRE_ATTACHMENT_RATE:.0%}",
)
m3.metric(
    "Expected Fire Attachments",
    f"{expected_multiline_fire:,.1f}",
)
st.caption(
    "Multiline fire attachments are estimated from auto production and are not used "
    "to increase the budget recommendation."
)


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
        .properties(title="Goal probability by budget", height=320)
    )
    probability_rule = (
        alt.Chart(
            pd.DataFrame(
                {"Selected Probability": [desired_probability_percent]}
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
    source_forecast_display = source_summary.copy()
    source_forecast_number_columns = [
        "Expected Leads",
        "Expected Quotes",
        "Expected Closed Policies",
        "Expected Auto Policies",
        "Expected Vehicle Items",
        "Expected Multiline Fire Attachments",
        "10th Percentile Vehicle Items",
        "Median Vehicle Items",
        "90th Percentile Vehicle Items",
        "Incremental Forecast-Period Cost",
    ]
    source_forecast_display[source_forecast_number_columns] = (
        source_forecast_display[source_forecast_number_columns].round(1)
    )
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
            "Assignment Hours",
            "Quote Rate",
            "Quote-to-Close Rate",
            "Lead-to-Close Rate",
            "Historical Cost Per Close",
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
    historical_display["Historical Cost Per Close"] = historical_display[
        "Historical Cost Per Close"
    ].round(2)
    historical_display = historical_display.rename(
        columns={
            "Cost Per Lead": "Cost Per Lead ($)",
            "Assignment Hours": "Average Assignment Hours",
            "Quote Rate": "Quote Rate (%)",
            "Quote-to-Close Rate": "Quote-to-Close Rate (%)",
            "Lead-to-Close Rate": "Lead-to-Close Rate (%)",
            "Historical Cost Per Close": "Historical Cost Per Close ($)",
        }
    )
    st.dataframe(historical_display, use_container_width=True, hide_index=True)



# =====================================================
# ASSUMPTIONS AND LIMITATIONS
# =====================================================

with st.expander("Model assumptions and limitations"):
    st.markdown(
        f"""
        <div class="assumption-box">
            <strong>Baseline calibration:</strong> The agency was already spending
            ${BASELINE_MONTHLY_BUDGET:,.0f} per month during the historical period.
            The model therefore anchors that budget to observed historical production.
        </div>
        """,
        unsafe_allow_html=True,
    )

    assumption_table = pd.DataFrame(
        {
            "Assumption": [
                "Historical vehicle pace",
                "Selected-period historical-pace forecast",
                "Vehicle items per auto policy",
                "Tracked-source auto share",
                "StateFarm.com monthly planning volume",
                "Multiline attachment rate",
                "Conversion scenario",
                "Paid-provider capacity assumption",
            ],
            "Value": [
                f"{HISTORICAL_MONTHLY_VEHICLE_PACE:.1f} per month",
                f"{historical_pace_forecast:.1f}",
                f"{VEHICLES_PER_AUTO_POLICY:.1f}",
                f"{AUTO_POLICY_SHARE:.1%}",
                f"{STATEFARM_MONTHLY_LEADS} leads",
                f"{MULTILINE_FIRE_ATTACHMENT_RATE:.0%}",
                conversion_label,
                (
                    f"Maximum {capacity_multiplier:.2f}× historical paid volume"
                    if apply_capacity_limits
                    else "Not applied"
                ),
            ],
        }
    )
    st.dataframe(assumption_table, use_container_width=True, hide_index=True)

    st.write(
        f"The provider-level records explain approximately "
        f"**${PAID_PROVIDER_RECORDED_MONTHLY_COST:,.0f} per month** of the established "
        f"$3,600 budget. Approximately **${UNMAPPED_MONTHLY_BUDGET:,.0f} per month** "
        "is not mapped to the three paid-provider CPL records. The model therefore "
        "uses CPL only for incremental spending above the baseline."
    )
    st.write(
        "The six tracked sources do not contain source-specific auto/fire splits. "
        "Their auto production is estimated using the agency-wide historical auto "
        "share of 73%."
    )
    st.write(
        "Additional paid spending affects only EverQuote, Smart Financial, and "
        "Insurance Quotes. Other agency production, StateFarm.com, referrals, and "
        "winbacks remain at their planning or historical monthly volumes."
    )
    st.write(
        "Provider capacity limits are planning assumptions because confirmed monthly "
        "maximums were not available. Adjust or disable them in the sidebar."
    )
    st.write(
        "The conversion scenarios adjust quote-to-close performance by 10% below, "
        "equal to, or 10% above the historical level."
    )
    st.write(
        "Assignment time is displayed for operational context but does not change "
        "conversion because the aggregate data does not establish causation."
    )
    st.warning(
        "This dashboard is a planning model, not a guarantee of future production."
    )