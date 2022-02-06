module app;

public import data_management;


/**
 * Generates the data for the paper.
 */
void main()
{
    auto nValues = [50];
    const numberOfRuns = 2000;
    auto kValues = iota(2, 9).array() ~ [10, 15, 20];
    
    TestCase[] testCases = [];
    foreach(n; nValues)
    {
        testCases ~= kValues
            .map!(k =>
                [
                    TestCase(n, k, PortfolioName.POWERS_OF_TWO, PickerName.OPTIMAL),
                    TestCase(n, k, PortfolioName.INITIAL_SEGMENT, PickerName.OPTIMAL),
                    TestCase(n, k, PortfolioName.EVENLY_SPREAD, PickerName.OPTIMAL),
                ]
                ~ (
                    (k<= 8) ?
                    [TestCase(n, k, PortfolioName.OPTIMAL, PickerName.OPTIMAL)] :
                    []
                )
            )()
            .array()
            .join();
    }

    // Generate data
    runDifferentRLSTests(testCases, numberOfRuns);
    evaluateDifferentRLSTests(testCases);

    // Tables 1 and 2
    testCases = [];
    nValues ~= [100];
    foreach(n; nValues)
    {
        foreach(k; kValues)
        {
            testCases ~=
                [
                    TestCase(n, k, PortfolioName.POWERS_OF_TWO, PickerName.OPTIMAL),
                    TestCase(n, k, PortfolioName.INITIAL_SEGMENT, PickerName.OPTIMAL),
                    TestCase(n, k, PortfolioName.EVENLY_SPREAD, PickerName.OPTIMAL),
                ]
                ~ (
                    ((k == 8 && n == 100) || k > 8) ?
                    [] :
                    [TestCase(n, k, PortfolioName.OPTIMAL, PickerName.OPTIMAL)]
                );
        }
    }
    generateInformation(testCases);
    
    // Figure 1
    calculateCumulativeOptimalPolicyData(50, 3);

    // Figure 2
    nValues = nValues[0 .. 1];
    PlotCase[] plotCases = [];
    foreach(n; nValues)
    {
        plotCases ~=
            [
                PlotCase([n], kValues, PortfolioName.POWERS_OF_TWO, PickerName.OPTIMAL, VisualizationType.VARY_K,
                    VisualizationDataSource.OPTIMAL),
                PlotCase([n], kValues, PortfolioName.INITIAL_SEGMENT, PickerName.OPTIMAL, VisualizationType.VARY_K,
                    VisualizationDataSource.OPTIMAL),
                PlotCase([n], kValues, PortfolioName.EVENLY_SPREAD, PickerName.OPTIMAL, VisualizationType.VARY_K,
                    VisualizationDataSource.OPTIMAL),
                PlotCase([n], kValues, PortfolioName.OPTIMAL, PickerName.OPTIMAL, VisualizationType.VARY_K,
                    VisualizationDataSource.OPTIMAL),
            ];
    }
    generateMultiplePlotData(plotCases);
}