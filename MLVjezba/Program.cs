using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLVjezba
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] amounts = new int[] { 3000, 200, 150, 200, 300, 250, 300,22, 2250, 400, 250, 400, 233, 0, 3, 5 };
            List<Withdrawl> withdrawls = amounts.Select(amount => new Withdrawl { Amount = amount }).ToList();

            var machineLearningContext = new MLContext();

            var estimator = machineLearningContext.Transforms.DetectIidSpike(
                   outputColumnName: nameof(Prediction.Output),
                   inputColumnName: nameof(Withdrawl.Amount),
                   confidence: 80,
                   pvalueHistoryLength: amounts.Length / 2);

            IDataView amountsData = machineLearningContext.Data.LoadFromEnumerable(withdrawls);
            IDataView transformedAmountsData = estimator.Fit(amountsData).Transform(amountsData);

            List<Prediction> predictions =
                machineLearningContext.Data.
                CreateEnumerable<Prediction>(transformedAmountsData, reuseRowObject: false).ToList();

            foreach(Prediction prediction in predictions)
            {
                double isAnomaly = prediction.Output[0];
                double originalValue = prediction.Output[1];
                double confidenceLevel = prediction.Output[2];

                Console.WriteLine($"{originalValue} \t {confidenceLevel} \t {isAnomaly}");
            }

            Console.ReadKey();
        }
    }

    class Withdrawl
    {
        public float Amount { get; set; }
    }

    class Prediction
    {
        [VectorType]
        public double[] Output { get; set; }
    }
}
