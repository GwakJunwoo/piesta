class Backtest:
    def __init__(self, start_date, end_date, capital, pipeline, data, slippage=0, commission=0):
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.pipeline = pipeline
        self.data = data
        self.slippage = slippage
        self.commission = commission

    def run_backtest(self):
        start_time = time.time()

        # Get assets to be traded
        assets = self.pipeline.screen(self.start_date)

        # Initialize portfolio holdings and performance
        portfolio = pd.DataFrame(index=[self.start_date], columns=assets, data=0)
        portfolio_value = pd.Series(index=[self.start_date], data=self.capital)

        # Loop over trading days
        for date in pd.date_range(start=self.start_date + pd.Timedelta(days=1), end=self.end_date, freq='D'):

            # Get pipeline output for current date and assets
            pipeline_output = self.pipeline.compute(date, assets=assets)

            # Calculate optimal portfolio weights
            objective = objective_functions.MaximizeReturns(pipeline_output['expected_returns'], pipeline_output['covariance'])
            constraints = []
            for i, lim in enumerate(pipeline_output['constraints']):
                constraints.append(constraint_functions.LimitWeight(asset=assets[i], limit=lim))
            problem = constraints_manager.ConstraintsManager(objective=objective, constraints=constraints)
            weights = problem.solve()

            # Adjust weights based on current holdings and available capital
            available_capital = portfolio_value[date]
            target_portfolio_value = available_capital * weights
            trades = target_portfolio_value - portfolio.iloc[-1] * portfolio_value[date]
            trades_cost = self.calculate_trades_cost(trades)
            trades = trades - trades_cost
            trades = self.apply_slippage(trades)
            trades = self.apply_commission(trades)
            trades_value = np.sum(trades * self.data.loc[date, assets])
            portfolio_value[date] = available_capital - trades_cost - trades_value
            portfolio.loc[date] = portfolio.iloc[-1] + trades / self.data.loc[date, assets]

        # Calculate portfolio performance
        returns = portfolio_value.pct_change().dropna()
        total_return = returns.iloc[-1]
        cagr = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe_ratio = sharpe_ratio(returns, 0)
        max_drawdown = max_drawdown(returns)

        # Print performance metrics
        print('Total Return: {:.2%}'.format(total_return))
        print('CAGR: {:.2%}'.format(cagr))
        print('Sharpe Ratio: {:.2f}'.format(sharpe_ratio))
        print('Max Drawdown: {:.2%}'.format(max_drawdown))
        print('Time Elapsed: {:.2f} seconds'.format(time.time() - start_time))

        return portfolio_value, returns, total_return, cagr, sharpe_ratio, max_drawdown

    def calculate_trades_cost(self, trades):
        return 0

    def apply_slippage(self, trades):
        return trades

    def apply_commission(self, trades):
        return trades
