# Volatility Surface Intelligence Engine

A Python + Streamlit research application for studying how implied volatility behaves across historical SPY options data and whether machine-learning models can learn that structure.

The project builds on the ideas from my **Quantitative Options Research Engine** by moving from live option-chain analysis into historical data, implied-volatility extraction, and out-of-sample machine learning.

The goal is to make the project understandable to someone who has never worked with volatility surfaces or machine learning, while still demonstrating concepts used in quantitative finance and computational research.

**Important**: This is a research and educational project. The historical dataset is large and the implied-volatility calculations are computationally expensive. This application is not a trading or execution system.


-----------------------------------------------------------------------------------------


**Why I recommend running this locally**


The application performs implied-volatility extraction using numerical root-finding.

Unlike simply loading an existing IV column, the application calculates IV from historical option quotes and then uses those results for feature engineering and machine-learning analysis.

The dataset can contain millions of option observations.

The general process looks like:

    Historical SPY Options
            ↓
    Quote Filtering
            ↓
        Market Mid
            ↓
    Implied Volatility Extraction
            ↓
    Feature Engineering
            ↓
    Machine Learning
            ↓
    Out-of-Sample Validation
            ↓
    BSM Pricing Analysis


The most computationally expensive part is the IV extraction.

For each option, the program has to numerically solve for the volatility that makes the Black-Scholes model reproduce the observed market price.

This uses Brent's root-finding method and therefore requires repeated pricing calculations for individual contracts.

Because of this, the full application can take several minutes to process large date ranges.

The deployed Streamlit version is useful for exploring the project, but the available computing resources are more limited than a local machine. This means that a large dataset or broad date range can take significantly longer to process in the hosted version.

### For the full experience, I recommend running the application locally.


Running locally allows you to:

    1. Process larger date ranges

    2. Change the filtering parameters

    3. Run the full IV extraction pipeline

    4. Compare different machine-learning models

    5. Experiment without waiting for the hosted environment

    6. Use the computing resources available on your own machine


I intentionally chose not to replace the expensive calculation with dozens
of precomputed datasets just to make the hosted version load faster.

The point of the project is to actually perform the numerical and
machine-learning workflow.


-----------------------------------------------------------------------------------------


**Running the application**


(ChatGPT'ed this part to follow usual GitHub repo lingo)


### 1. Clone the repository

    >git clone <your-repository-url>
    >cd volatility-surface-intelligence-engine


### 2. Create a virtual environment

Windows

    >python -m venv .venv
    >.venv\Scripts\activate

macOS / Linux

    >python -m venv .venv
    >source .venv/bin/activate


Python virtual environments:

https://docs.python.org/3/library/venv.html


### 3. Install dependencies

    >pip install -r requirements.txt


### 4. Download the historical SPY data

The project uses historical SPY option data stored in Parquet format.

Download the years you want to work with using:

    >python -m src.download_spy --years 2024


For multiple years:

    >python -m src.download_spy --years 2021 2022 2023 2024 2025


You do not need to download every year if you only want to experiment
with a smaller dataset.


### 5. Start Streamlit

    >streamlit run app.py


The application should then open in your browser.


-----------------------------------------------------------------------------------------


**Things that I want to talk about:**


1. What is this project?

2. Why I recommend running this locally

3. Running the application

4. Historical SPY options data

5. What the application does

6. Market data vs model data

7. Implied volatility

8. Why Brent root-finding is used

9. Bad quotes and liquidity filtering

10. Feature engineering

11. Volatility surfaces

12. Machine learning approach

13. Gradient Boosting

14. MLP Neural Network

15. Chronological train/validation/test splitting

16. Out-of-sample prediction

17. MAE and RMSE

18. Residual analysis

19. Downstream Black-Scholes pricing error

20. Application workflow

21. Performance and computational considerations

22. Caching

23. Project structure

24. Testing

25. Learning resources

26. Future improvements

27. Limitations

28. Disclaimer


-----------------------------------------------------------------------------------------


**What is this project?**


The Volatility Surface Intelligence Engine is a historical options
research application built with:

    PYTHON for quantitative calculations

    NUMPY for numerical computation

    PANDAS for data processing

    SCIPY for numerical root-finding

    SCIKIT-LEARN for machine learning

    PYARROW for Parquet data

    STREAMLIT for the interactive interface

    PLOTLY for visualization


The application asks a slightly different question from my first
options project.

The first project asks:

    "Given the market data, what does the pricing model tell us?"


This project asks:

    "Can we learn the structure of implied volatility from historical
    option-market data?"


The research pipeline is:

    Historical Option Data
            ↓
    Quote Cleaning
            ↓
        Market Mid
            ↓
    Implied Volatility
            ↓
    Feature Engineering
            ↓
    Train / Validation / Test
            ↓
    Machine Learning
            ↓
    Predicted IV
            ↓
    Residual Analysis
            ↓
    Black-Scholes Validation


The project is therefore both a volatility-surface research tool and a
demonstration of how machine learning can be applied to structured
financial data.


-----------------------------------------------------------------------------------------


**Historical SPY options data**


This project uses historical SPY option-market observations rather than
the current option chain.

The data contains information such as:

    -Option date

    -Expiration

    -Strike

    -Call / Put

    -Bid

    -Ask

    -Volume

    -Open Interest

    -Implied volatility

    -Underlying information


The data is stored in Parquet format.

Parquet is useful here because the dataset is much larger than a typical
CSV used for a small data-science project.

Instead of loading unnecessary columns and processing everything as text,
Parquet allows the application to work with structured numerical data
more efficiently.


-----------------------------------------------------------------------------------------


**What the application does**


The application takes historical option-market data and processes it
through several stages.

It can work with:

    -Underlying price

    -Strike

    -Expiration

    -Call / Put

    -Bid

    -Ask

    -Volume

    -Open Interest

    -Market spread


It then calculates or displays:

    -Market Mid

    -Calculated IV

    -Log-moneyness

    -Time to expiration

    -Predicted IV

    -Prediction residuals

    -MAE

    -RMSE

    -Black-Scholes price differences


The application also compares:

    Gradient Boosting

    MLP Neural Network


and visualizes the resulting implied-volatility surface.


-----------------------------------------------------------------------------------------


**Market data vs model data**


This is one of the ideas carried over from my Quantitative Options
Research Engine.

There are two fundamentally different kinds of numbers.


##Market values


These come from the historical market-data source:

    Bid

    Ask

    Volume

    Open Interest


##Model values


These are calculated or inferred by mathematical models:

    Market Mid

    Calculated IV

    Predicted IV

    BSM Price

    Prediction Residual


Keeping these categories separate is important.

A market observation is not the same thing as a model estimate.

The project therefore treats the historical quote as the starting point
and calculates implied volatility from that quote rather than treating
the supplied IV as unquestionable ground truth.


-----------------------------------------------------------------------------------------


**Implied volatility**


Implied volatility, or IV, works backwards from an option price.

Black-Scholes naturally answers:

    Given volatility
            ↓
    Calculate option price


For implied volatility, we reverse the problem:

    Given market price
            ↓
    Find volatility
            ↓
    That makes the model price match the market price


The application therefore solves:

    Black-Scholes Price(IV) - Market Price = 0


using SciPy's Brent root-finding method.


The calculated IV becomes the target that the machine-learning models
attempt to learn.


The important distinction is:

    Bid / Ask / Mid
            ↓
    Market information

    Calculated IV
            ↓
    Value inferred from the market using a model


IV is therefore not directly observed in the same way as Bid or Ask.


-----------------------------------------------------------------------------------------


**Why Brent root-finding is used**


Black-Scholes naturally answers:

    Given volatility → calculate price


We want to solve:

    Given price → calculate volatility


There is no simple closed-form formula for standard Black-Scholes
implied volatility.

The application therefore solves:

    ModelPrice(IV) - MarketPrice = 0


using SciPy's `brentq()` root finder.

Brent's method searches for a root inside a bracket where the function
changes sign.


Resource:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html


##Why brentq() can fail


Not every market price corresponds to a valid IV.

For example:

    Impossible market price
            ↓
    No volatility can reproduce it
            ↓
    No mathematical root
            ↓
    brentq cannot converge


The application therefore checks:

    1. Are the inputs valid?

    2. Is the market price finite?

    3. Is the market price inside theoretical bounds?

    4. Is the root bracketed?

    5. Does Brent actually converge?


If not, the application does not invent an IV.

Instead, the observation can be marked as unavailable or excluded from
the relevant calculation.


-----------------------------------------------------------------------------------------


**Bad quotes and liquidity filtering**


Historical market data is messy.

The dataset can contain:

    1. Missing bids

    2. Missing asks

    3. Zero bids

    4. Zero asks

    5. Very wide spreads

    6. Missing volume

    7. Missing open interest

    8. Extreme values

    9. Prices that violate theoretical bounds

    10. Other observations that are not useful for IV extraction


Therefore the application filters the data before using observations
for the research pipeline.


For example:

    Zero quote

        Bid = $0
        Ask = $0


This should not automatically become:

        Mid = $0


A crossed market should also be rejected:

        Bid = $10
        Ask = $9


A wide spread can also make an observation less useful:

        Bid = $10
        Ask = $20


The mathematical midpoint is $15, but the quote is extremely wide.


The principle is:

    A number appearing in a data feed does not automatically make it a
    trustworthy market observation.


-----------------------------------------------------------------------------------------


**Feature engineering**


The machine-learning models do not simply receive the raw option price.

The application creates features describing each contract.


These include:

    -Log-moneyness

    -Time to expiration

    -Option type

    -Spot price

    -Strike price

    -Trading volume

    -Open interest

    -Bid/Ask spread

    -Risk-free rate

    -Dividend yield


The target variable is:

    Calculated Implied Volatility


The goal is therefore:

    Option characteristics
            ↓
    Machine Learning Model
            ↓
    Predicted IV


-----------------------------------------------------------------------------------------


**What is log-moneyness?**


Moneyness describes the relationship between the underlying price and
the strike price.

A simple version is:

    S / K


where:

    S = underlying price

    K = strike price


This project uses log-moneyness:

    ln(S / K)


This gives the model a numerical representation of whether an option is
above, near, or below the underlying price while providing a useful
continuous feature for the machine-learning models.


-----------------------------------------------------------------------------------------


**Training, validation, and test data**


The project does not randomly shuffle all historical observations before
training.

Instead, it uses chronological splits.

Conceptually:

    Earlier data
         ↓
       TRAIN
         ↓
    VALIDATION
         ↓
       TEST
         ↓
    Later unseen data


The training set is used to fit the model.

The validation set is used to compare and tune models.

The test set is held out until evaluation.


This matters because financial data is time-dependent.

A random split could allow information from later periods to influence
the training process and produce an overly optimistic evaluation.


The goal is therefore to ask:

    "How well does the model perform on historical observations that it
    did not see during training?"


-----------------------------------------------------------------------------------------


**Gradient Boosting**


Gradient Boosting is one of the machine-learning models used in this
project.

The basic idea is to build a sequence of decision trees.

Conceptually:

    First model
        ↓
    Find errors
        ↓
    Next model focuses on those errors
        ↓
    Repeat
        ↓
    Final prediction


Gradient Boosting is useful for this type of structured tabular data
because it can learn nonlinear relationships and interactions between
features.


For example, the relationship between:

    Moneyness

    Time to expiration

    Option type

    Liquidity


does not have to be a simple straight line.


-----------------------------------------------------------------------------------------


**MLP Neural Network**


The project also uses an MLP, or Multi-Layer Perceptron, neural network.


Conceptually:

    Input Features
          ↓
    Hidden Layer
          ↓
    Hidden Layer
          ↓
    Output
          ↓
    Predicted IV


The MLP provides a different approach to learning the relationship
between option characteristics and implied volatility.


The purpose of including both models is not to assume that one model
must be better.

It allows the models to be compared under the same dataset, features,
and chronological evaluation process.


-----------------------------------------------------------------------------------------


**Out-of-sample prediction**


"Out-of-sample" simply means that the model is making predictions on
data it did not use to learn its parameters.


For example:

    Training Data
         ↓
    Model learns
         ↓
    Model sees new Test Data
         ↓
    Prediction


This is important because a model can perform extremely well on the data
it was trained on without actually generalizing to new observations.


The project therefore focuses on out-of-sample performance rather than
only looking at training accuracy.


-----------------------------------------------------------------------------------------


**MAE and RMSE**


The application evaluates the models using metrics such as:

    MAE

    RMSE


##MAE

Mean Absolute Error measures the average absolute difference between the
actual IV and predicted IV.


Conceptually:

    Actual IV - Predicted IV
            ↓
        Absolute value
            ↓
        Average


For example, an MAE of:

    0.025


roughly corresponds to an average absolute difference of 2.5 volatility
points.


##RMSE

Root Mean Squared Error works similarly but gives more weight to larger
errors.


Conceptually:

    Error
      ↓
    Square
      ↓
    Average
      ↓
    Square root


This makes RMSE useful for identifying whether a model has some
particularly large prediction errors.


Both metrics are reported because they provide slightly different views
of model performance.


-----------------------------------------------------------------------------------------


**Residuals**


A residual is the difference between the actual value and the model's
prediction.


In this project:

    Residual = Actual IV - Predicted IV


For example:

    Actual IV = 0.30

    Predicted IV = 0.25

    Residual = 0.05


A positive residual means the model underpredicted IV.

A negative residual means the model overpredicted IV.

A residual of zero means the prediction exactly matched the calculated IV.


Residuals are useful because an average error alone can hide patterns.


-----------------------------------------------------------------------------------------


**Residual analysis**


The application visualizes residuals to look for patterns in the
model's errors.


For example, we can ask:

    Does the model consistently underpredict high-volatility options?

    Do errors increase for extreme moneyness?

    Do errors become larger near expiration?

    Are there large outliers?

    Does the error spread change across the volatility range?


A good residual analysis is therefore more informative than simply
reporting one number such as MAE.


One of the patterns worth investigating is whether prediction errors
become larger at higher levels of implied volatility.


This can indicate that the model is not learning every part of the
volatility surface equally well.


-----------------------------------------------------------------------------------------


**The implied-volatility surface**


The application visualizes implied volatility across dimensions such
as:

    Log-moneyness

    Time to expiration

    Implied volatility


Conceptually:

                 IV
                  ^
                  |
              ___/----
          ___/
      ___/
  ___/
  ---------------------------->
       Log-moneyness


The actual application uses a three-dimensional visualization so that
the relationship between moneyness, maturity, and volatility can be
explored interactively.


This is an important step beyond calculating IV for individual options.

Instead of asking:

    "What is the IV of this option?"


we can ask:

    "What does the overall structure of implied volatility look like?"


-----------------------------------------------------------------------------------------


**Downstream Black-Scholes pricing error**


The project does not stop at predicting implied volatility.


An IV prediction is useful only if we understand what that error means
for the option price.


The application therefore feeds both calculated IV and predicted IV
through Black-Scholes.


Conceptually:

    Calculated IV
          ↓
    Black-Scholes
          ↓
    Reference Price


and:

    Predicted IV
          ↓
    Black-Scholes
          ↓
    ML Price


The difference between the two gives a downstream pricing error.


This asks a more practical question:

    "If the model gets volatility slightly wrong, how much does that
    actually change the estimated option price?"


This is useful because an IV error and a price error are not necessarily
equivalent across every option.


-----------------------------------------------------------------------------------------


**Model comparison**


The project currently compares:

    1. Gradient Boosting

    2. MLP Neural Network


The models are evaluated using the same general process:

    Same historical dataset
            ↓
    Same feature set
            ↓
    Same chronological split
            ↓
    Train model
            ↓
    Predict unseen data
            ↓
    MAE / RMSE
            ↓
    Residual analysis
            ↓
    Downstream BSM pricing error


The goal is not to assume that one algorithm is always better.

The goal is to measure how different models behave on this particular
dataset and under the same experimental setup.


-----------------------------------------------------------------------------------------


**Application workflow**


The general pipeline is:

    Historical SPY Data
            │
            ▼
    Quote Validation
            │
            ▼
    Market Mid
            │
            ▼
    No-Arbitrage Checks
            │
            ▼
    Implied Volatility
            │
            ▼
    Feature Engineering
            │
            ▼
    Chronological Split
            │
       ┌────┴────┐
       ▼         ▼
    Gradient     MLP
    Boosting    Neural Net
       │         │
       └────┬────┘
            ▼
    Out-of-Sample IV
            │
       ┌────┴────┐
       ▼         ▼
    Residual   BSM Price
    Analysis   Comparison
       │         │
       └────┬────┘
            ▼
    Interactive Streamlit UI


-----------------------------------------------------------------------------------------


**Performance and computational considerations**


An inefficient approach would attempt to perform every operation using
Python-level DataFrame loops.

The project therefore uses NumPy-based calculations where possible.


Conceptually:

    Row by row:

    Option 1 → calculate
    Option 2 → calculate
    Option 3 → calculate


versus:

    [Option 1, Option 2, Option 3, ...]
                    ↓
              NumPy calculation


This reduces unnecessary Python-level looping.


However, implied-volatility extraction is different.

Each option has its own market price and therefore its own root-finding
problem.

The IV solver still has to work contract-by-contract.


This is one of the main reasons that the historical research pipeline
takes considerably more computation than the final visualization.


-----------------------------------------------------------------------------------------


**Caching**


Streamlit reruns the application when widgets change.

Without caching, expensive calculations or data-loading operations could
be repeated unnecessarily.


The project therefore uses Streamlit caching where appropriate.


For example:

    @st.cache_data


can be used for serializable data-processing results.


Streamlit documentation:

https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data


Caching is useful here, but it does not completely remove the underlying
computational cost of extracting IV from millions of historical option
observations.


-----------------------------------------------------------------------------------------


**Project structure**


    volatility-surface-intelligence-engine/
    │
    ├── app.py
    │
    ├── src/
    │   ├── download_spy.py
    │   │
    │   ├── data_loader.py
    │   │
    │   ├── models/
    │   │
    │   └── ...
    │
    ├── data/
    │   └── raw/
    │       └── spy/
    │
    ├── tests/
    │
    ├── requirements.txt
    ├── README.md
    └── .gitignore


##app.py

Streamlit interface and overall application workflow.


##download_spy.py

Downloads the historical SPY option data required by the application.


##data_loader.py

Loads and prepares the historical option data and underlying information.


##models/

Contains the machine-learning model implementations and related logic.


##data/

Contains the historical Parquet data used by the research pipeline.


-----------------------------------------------------------------------------------------


**Testing**


The application should be tested at both the numerical and
machine-learning levels.


##Numerical tests


The underlying pricing and IV calculations should be tested using:

    1. Known Black-Scholes benchmarks

    2. Valid implied-volatility round trips

    3. No-arbitrage bounds

    4. Invalid market prices

    5. Missing quotes

    6. Zero quotes

    7. Crossed markets

    8. Wide spreads

    9. IV convergence failures


Expected behavior includes:

    No application crash

    Invalid observations rejected or flagged

    IV shown as unavailable when appropriate


##Machine-learning tests


The ML pipeline should also be evaluated using:

    1. Chronological train/test splitting

    2. MAE

    3. RMSE

    4. Residual analysis

    5. Out-of-sample predictions

    6. Downstream Black-Scholes pricing error


The goal is to test both whether the model predicts IV accurately and
whether those prediction errors have meaningful effects on option
pricing.


-----------------------------------------------------------------------------------------


**Learning resources**


##Options fundamentals

The Options Industry Council is a useful starting point for:

    Calls

    Puts

    Strike prices

    Expiration

    Premiums

https://www.optionseducation.org/


##Option pricing

https://www.optionseducation.org/optionsoverview/options-pricing


##Greeks

https://www.optionseducation.org/advancedconcepts/understanding-options-greeks


##Akuna Capital Options 101

This was particularly useful for understanding options from a
market-making perspective.

Topics include:

    Options basics

    Time premium

    Put-call parity

    Theoretical values / Theos

    Greeks

    Volatility

    Vega

    Delta hedging

https://akunacapital.teachable.com/p/options101


##Black-Scholes

Fischer Black and Myron Scholes,

" The Pricing of Options and Corporate Liabilities. "

https://www.jstor.org/stable/1831029


##Numerical methods

SciPy's brentq documentation:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html


NumPy:

https://numpy.org/doc/


Pandas:

https://pandas.pydata.org/docs/


Scikit-learn:

https://scikit-learn.org/stable/


Streamlit:

https://docs.streamlit.io/


-----------------------------------------------------------------------------------------


**Future improvements**


1. SVI volatility-surface calibration

Instead of predicting IV independently for each observation, fit a
structured volatility surface across:

    Strike

    Maturity

    Volatility


This would allow the machine-learning predictions to be compared
against a more traditional volatility-surface model.


2. SABR volatility model

Add SABR calibration for modeling volatility smiles and term structures.


3. Better interest-rate data

The current research pipeline uses a simplified risk-free rate assumption.

A more advanced implementation could use historical maturity-matched
interest-rate data.


4. Better dividend data

A future version could incorporate historical dividend information or
a more detailed dividend term structure.


5. Additional machine-learning models

Future experiments could include:

    Random Forest

    XGBoost

    LightGBM

    Other neural-network architectures


The important part would be comparing them using the same chronological
evaluation framework rather than adding models simply for the sake of
having more models.


6. Better volatility-surface validation

Future work could investigate whether the predicted surface satisfies
additional financial and numerical constraints.


7. More downstream analysis

The predicted IV could be used to study:

    Option-price errors

    Greek errors

    Surface reconstruction

    Missing-quote reconstruction

    Stability across market regimes


-----------------------------------------------------------------------------------------


**Limitations**


Historical option data is not perfect.

It can contain:

    Missing observations

    Wide spreads

    Stale quotes

    Illiquid contracts

    Inconsistent fields

    Extreme values


The application therefore depends heavily on the quality of the
underlying dataset and the filtering assumptions.


##Implied volatility is model-dependent


There is no completely model-independent "true IV."


IV depends on:

    1. Market price

    2. Pricing model

    3. Interest rate

    4. Dividend assumption

    5. Time convention

    6. Numerical method


Two systems can therefore produce different IVs from the same market
data if their assumptions differ.


##Machine learning limitations


A model that performs well on historical data does not automatically
mean that it will perform well on future market data.

The relationship between option characteristics and implied volatility
can change across different market environments.

This is why the project uses chronological out-of-sample testing rather
than relying only on training performance.


##Computational limitations


The full historical dataset can require significant CPU and memory.

The hosted Streamlit application may therefore take considerably longer
than a local machine when processing a large date range.


-----------------------------------------------------------------------------------------


**Disclaimer**


Again, if you've read this far then you should probably realize that:


    1. This project is for education, research, and software-development
       purposes only.


    2. It is not investment advice.


    3. It is not a trading recommendation.


    4. It is not an execution system.


    5. Predicted implied volatilities, theoretical prices, and other
       outputs depend on model assumptions and the quality of the
       underlying historical data.


    6. Machine-learning predictions are not guaranteed to generalize to
       future market conditions.


    7. Do not use this application as the sole basis for financial
       decisions.


**This project is simply an attempt to take the options-pricing concepts
from my first project and investigate what happens when historical
market data, numerical methods, and machine learning are combined.**


-----------------------------------------------------------------------------------------


**Why this project matters for quantitative finance**


The project demonstrates a small version of a larger quantitative
research workflow:


    Historical Market Data
            ↓
    Data Cleaning
            ↓
    Mathematical Model
            ↓
    Numerical Methods
            ↓
    Feature Engineering
            ↓
    Machine Learning
            ↓
    Out-of-Sample Validation
            ↓
    Error Analysis
            ↓
    Financial Interpretation
            ↓
    Research


The interesting part is not simply training a machine-learning model.


The interesting part is understanding:


    1. What data went into the calculation?

    2. What assumptions were made?

    3. How was implied volatility obtained?

    4. How was the model evaluated?

    5. Where does the model make mistakes?

    6. Do those mistakes actually matter for option pricing?


That is the part of the project I found most interesting.

The machine-learning model is only one part of the research process.