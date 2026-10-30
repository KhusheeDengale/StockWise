"""Page 6: AI/ML Lab — Academic explanations of all algorithms used in StockWise.
"""
import streamlit as st
from ui.components import render_info_box, render_page_header


def render_aiml_lab_page():
    """Render the AI/ML Lab with algorithm explanations."""
    render_page_header(
        title="🧠 AI/ML Lab — How StockWise Works",
        subtitle="Understand every algorithm in plain language with academic detail.",
    )

    render_info_box(
        "What is this page?",
        "This is an <b>educational walkthrough</b> of every algorithm used in StockWise. "
        "Each section explains the method, its purpose, and how it is applied to your stationery inventory data."
    )

    tab_forecast, tab_eval, tab_features, tab_fuzzy, tab_ga, tab_greedy = st.tabs([
        "📈 Forecasting Models",
        "📊 Model Evaluation",
        "🔧 Feature Engineering",
        "🔴 Fuzzy Logic",
        "🧬 Genetic Algorithm",
        "📦 Greedy Baseline",
    ])

    with tab_forecast:
        st.markdown("### 📈 Time Series Forecasting Models")
        st.markdown("""
StockWise uses **4 forecasting models** to predict future daily demand for each store-product pair.
Each model is trained only on historical data and tested on unseen future data.

---

#### 1. Seasonal Naive
**What it does:** Copies last week's sales pattern into the future.

**How it works:**
- For each day to forecast, it looks at the same day of the week from the last complete week.
- Example: To predict next Monday's sales, it uses last Monday's sales.

**When it works best:** Products with strong weekly seasonality (e.g., higher sales on weekdays for office supplies).

**Limitation:** Cannot capture trends or respond to sudden changes.

---

#### 2. Moving Average (7-day)
**What it does:** Averages the last 7 days and uses that as the forecast for all future days.

**Formula:**
$$\\hat{y}_{t+h} = \\frac{1}{7} \\sum_{i=1}^{7} y_{t-i+1}$$

**When it works best:** Stable products without strong trends.

**Limitation:** Ignores seasonality and trends; all forecast days get the same value.

---

#### 3. Linear Regression
**What it does:** Fits a line through engineered features (lags, rolling averages, day-of-week) to predict demand.

**Features used:**
- `lag_7`, `lag_14`, `lag_28` — Demand 7, 14, 28 days ago
- `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` — Rolling averages
- `rolling_std_7` — Rolling standard deviation
- `day_sin`, `day_cos` — Cyclical encoding of day-of-week

**When it works best:** Products with identifiable patterns in lag features.

---

#### 4. ARIMA (Auto-Regressive Integrated Moving Average)
**What it does:** Models the time series as a combination of past values (AR), differencing (I), and past errors (MA).

**Order selection:** Tries a grid of (p, d, q) values and selects the one with the lowest AIC score.

**When it works best:** Series with trends and autocorrelation.

**Limitation:** Can be slow for long series; may not capture complex seasonality.
""")

    with tab_eval:
        st.markdown("### 📊 Chronological Backtesting & Model Selection")
        st.markdown("""
#### How Models are Evaluated

StockWise uses **strict chronological backtesting** — no information from the future is ever used during training or validation.

**Data Split:**
| Segment | Proportion | Purpose |
|---------|-----------|---------|
| Training | 70% | Model fitting |
| Validation | 15% | Model comparison & winner selection |
| Test | 15% | Final held-out verification |

#### Non-Overlapping Block Evaluation
- The validation and test segments are divided into blocks of length H (the forecast horizon).
- For each block, the model uses only data before the block to make predictions.
- The forecast origin advances by H days for each new block.

#### Metrics Used
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean Absolute Error | Average units of error per day |
| **RMSE** | Root Mean Square Error | Penalizes large errors more |
| **WAPE** | Weighted Absolute Percentage Error | Error as % of total demand |
| **R²** | Coefficient of Determination | How much variance is explained |

#### Winner Selection Rule
1. Sort models by **validation WAPE** (lower is better).
2. Tie-break: lower MAE → lower RMSE → simpler model.
3. Simplicity ranking: Seasonal Naive > Moving Average > Linear Regression > ARIMA.
4. The winner is refitted on **all available history** and used for the final forecast.
5. Test metrics are reported but **never change the winner**.
""")

    with tab_features:
        st.markdown("### 🔧 Feature Engineering for Linear Regression")
        st.markdown("""
Linear Regression needs handcrafted features to capture temporal patterns.

#### Feature Set
| Feature | Description |
|---------|-------------|
| `lag_7` | Demand 7 days ago |
| `lag_14` | Demand 14 days ago |
| `lag_28` | Demand 28 days ago |
| `rolling_mean_7` | Mean demand over last 7 days |
| `rolling_mean_14` | Mean demand over last 14 days |
| `rolling_mean_28` | Mean demand over last 28 days |
| `rolling_std_7` | Standard deviation of demand over last 7 days |
| `day_sin` | sin(2π × day_of_week / 7) — cyclical encoding |
| `day_cos` | cos(2π × day_of_week / 7) — cyclical encoding |

#### Feature Selection Process
1. Compute Pearson correlation of each feature with the target.
2. Remove features with |correlation| < 0.05 (too weak).
3. Check for collinearity — if two features correlate > 0.90, keep the one with higher target correlation.
4. Verify on validation set — ensure selected features improve MAE.
""")

    with tab_fuzzy:
        st.markdown("### 🔴 Mamdani Fuzzy Inference System (FIS)")
        st.markdown("""
#### Why Fuzzy Logic?
Traditional rules (if stock < 10, reorder) are brittle. Fuzzy logic handles **vagueness and uncertainty**
by using degree-of-membership instead of crisp boundaries.

#### Three Inputs → One Output

| Input Variable | Range | Linguistic Terms |
|---------------|-------|-----------------|
| **Coverage Ratio** | [0, 3] | Low, Medium, High |
| **Lead-Time Pressure** | [0, 1] | Low, Medium, High |
| **Forecast Uncertainty** | [0, 1] | Low, Medium, High |

| Output Variable | Range | Linguistic Terms |
|----------------|-------|-----------------|
| **Reorder Urgency** | [0, 100] | Low, Moderate, High, Critical |

#### Input Formulas
- **Coverage Ratio** = Current Stock / Demand During Lead Time
- **Lead-Time Pressure** = Lead Time Days / Planning Horizon
- **Forecast Uncertainty** = Validation MAE / Mean Daily Training Demand (clipped to [0, 1])

#### Rule Base: 27 Rules (3 × 3 × 3)
All combinations of Low/Medium/High for each input, each mapping to a consequent urgency level.

#### Inference Process
1. **Fuzzification** — compute membership degrees for each input.
2. **Rule evaluation** — AND = min, Implication = min (Mamdani).
3. **Aggregation** — combine fired rule outputs using max.
4. **Defuzzification** — Centroid method to get a crisp score [0, 100].
""")

    with tab_ga:
        st.markdown("### 🧬 Genetic Algorithm (GA) Optimizer")
        st.markdown("""
#### The Optimization Problem
Given N products, a budget, and forecasted demand, find order quantities that minimize:
$$\\text{Objective} = 0.55 \\times \\text{Shortage} + 0.20 \\times \\text{Excess} + 0.15 \\times \\text{Cost} + 0.10 \\times \\text{CoverageImbalance}$$

Subject to:
- Total cost ≤ Budget
- Each order quantity is a multiple of the product's OrderMultiple
- Current stock + order ≤ Storage capacity

#### GA Parameters
| Parameter | Value |
|-----------|-------|
| Population Size | 40 |
| Generations | 80 |
| Tournament Size | 3 |
| Crossover Probability | 0.8 |
| Mutation Probability | 0.1 |
| Elite Count | 2 |
| Early Stopping Patience | 20 |

#### Process
1. **Initialize** — Create 40 chromosomes (one zero-order, one greedy, rest random).
2. **Evaluate** — Compute objective for each chromosome.
3. **Select** — Tournament selection of parents.
4. **Crossover** — Single-point crossover (80% chance).
5. **Mutate** — Per-gene mutation (10% chance): zero-out, add, or subtract one multiple.
6. **Repair** — Ensure budget and storage constraints are met.
7. **Repeat** until convergence or 80 generations.

#### Invariant Guarantee
The GA solution is guaranteed to be ≤ the greedy baseline objective value. If not, the greedy solution is used.
""")

    with tab_greedy:
        st.markdown("### 📦 Greedy Baseline Optimizer")
        st.markdown("""
#### How Greedy Works
1. Sort products by fuzzy urgency score (highest urgency first).
2. For each product in order:
   a. Compute the "shortage gap" = forecasted demand − current stock.
   b. Round up to the nearest OrderMultiple.
   c. Check budget and storage constraints.
   d. Allocate as much as possible within constraints.
3. Repeat until budget is exhausted or all products are covered.

#### Why Greedy is a Baseline
- Greedy is fast (O(N log N) sorting + O(N) allocation).
- It does not explore trade-offs between products.
- The GA uses greedy as a seeded individual and is guaranteed to match or beat it.

#### Inventory Simulation
For each product, a day-by-day simulation runs:
- Stock starts at CurrentStock.
- On lead-time day, the ordered quantity arrives.
- Each day, forecasted demand is subtracted.
- Shortage = max(0, demand − available stock) summed over all days.
- Ending stock = stock at end of horizon.
""")
