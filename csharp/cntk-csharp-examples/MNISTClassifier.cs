using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Emgu.CV.Structure;
namespace CNTK.CSTrainingExamples
{
    /// <summary>
    /// This class shows how to build and train a classifier for handwritting data (MNIST).
    /// For more details, please follow a serial of tutorials below:
    /// https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103A_MNIST_DataLoader.ipynb
    /// https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103B_MNIST_LogisticRegression.ipynb
    /// https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103C_MNIST_MultiLayerPerceptron.ipynb
    /// https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb
    /// </summary>
    public class MNISTClassifier
    {
        /// <summary>
        /// Execution folder is: CNTK/x64/BuildFolder
        /// Data folder is: CNTK/Tests/EndToEndTests/Image/Data
        /// </summary>
        public static string ImageDataFolder = "../../../../../data/";

        /// <summary>
        /// Train and evaluate a image classifier for MNIST data.
        /// </summary>
        /// <param name="device">CPU or GPU device to run training and evaluation</param>
        /// <param name="useConvolution">option to use convolution network or to use multilayer perceptron</param>
        /// <param name="forceRetrain">whether to override an existing model.
        /// if true, any existing model will be overridden and the new one evaluated. 
        /// if false and there is an existing model, the existing model is evaluated.</param>
        public static void TrainAndEvaluate(DeviceDescriptor device, bool useConvolution, bool forceRetrain)
        {
            var featureStreamName = "features";
            var labelsStreamName = "labels";
            var classifierName = "classifierOutput";

            Function classifierOutput;
            int[] imageDim = useConvolution ? new int[] { 28, 28, 1 } : new int[] { 784 };
            int imageSize = 28 * 28;
            int numClasses = 10;

            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featureStreamName, imageSize), new StreamConfiguration(labelsStreamName, numClasses) };

            string modelFile = useConvolution ? "MNISTConvolution.model" : "MNISTMLP.model";

            // If a model already exists and not set to force retrain, validate the model and return.
            //prepare vars to accept results
            List<List<float>> X = new List<List<float>>();
            List<float> Y = new List<float>();
            if (File.Exists(modelFile) && !forceRetrain)
            {
                var minibatchSourceExistModel = MinibatchSource.TextFormatMinibatchSource(
                    Path.Combine(ImageDataFolder, "MINST-TestData.txt"), streamConfigurations);

                //Model validation       
                ValidateModel(modelFile, minibatchSourceExistModel, imageDim, numClasses, featureStreamName, labelsStreamName,
                    classifierName, device, 1000, X, Y, useConvolution);

                //show image classification result
                showResult(X, Y);
                return;
            }

            // build the network
            var input = CNTKLib.InputVariable(imageDim, DataType.Float, featureStreamName);

            if (useConvolution)
            {
                var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);
                classifierOutput = CreateConvolutionalNeuralNetwork(scaledInput, numClasses, device, classifierName);
            }
            else
            {
                // For MLP, we like to have the middle layer to have certain amount of states.
                int hiddenLayerDim = 200;
                var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);
                classifierOutput = CreateMLPClassifier(device, numClasses, hiddenLayerDim, scaledInput, classifierName);
            }

            var labels = CNTKLib.InputVariable(new int[] { numClasses }, DataType.Float, labelsStreamName);
            //LOss and Eval functions
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError");

            // prepare training data
            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(
                Path.Combine(ImageDataFolder, "MINST-TrainData.txt"), streamConfigurations, MinibatchSource.InfinitelyRepeat);

            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            // set per sample learning rate
            var learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.003125, 1);

            IList<Learner> parameterLearners = new List<Learner>()
                {
                    Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample)
                };

            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            //
            const uint minibatchSize = 64;
            int outputFrequencyInMinibatches = 100, i = 0;
            int epochs = 3;
            while (epochs > 0)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { input, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(arguments, device);
                //
                TestHelper.PrintTrainingProgress(trainer, i++, outputFrequencyInMinibatches);

                // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
                // Batching will not end. Each time minibatchSource completes an sweep (epoch),
                // the last minibatch data will be marked as end of a sweep. We use this flag
                // to count number of epochs.
                if (TestHelper.MiniBatchDataIsSweepEnd(minibatchData.Values))
                {
                    epochs--;
                }
            }

            // save the trained model
            classifierOutput.Save(modelFile);

            // validate the model
            var minibatchSourceNewModel = MinibatchSource.TextFormatMinibatchSource(
                Path.Combine(ImageDataFolder, "MINST-TestData.txt"), streamConfigurations, MinibatchSource.InfinitelyRepeat);

            //Model validation       
            ValidateModel(modelFile, minibatchSourceNewModel,imageDim, numClasses, featureStreamName, labelsStreamName, 
                classifierName, device,1000, X,Y, useConvolution);
            
            //show image classification result
            showResult(X,Y);
        }

        private static void showResult(List<List<float>> X, List<float> Y)
        {
            for (int i = 0; i < X.Count; i++)
            {
                var img = X[i];
                var result = $"Outup = {Y[i]}";
                Bitmap bmp = TestHelper.ArrayToImg(28, 28, img);
                var emgImg = new Emgu.CV.Image<Bgr, byte>(bmp);
                var resizedImg = emgImg.Resize(250, 250, Emgu.CV.CvEnum.Inter.Nearest);
                // show output
                Emgu.CV.UI.ImageViewer.Show(resizedImg, result);
            }
        }

        private static Function CreateMLPClassifier(DeviceDescriptor device, int numOutputClasses, int hiddenLayerDim,
            Function scaledInput, string classifierName)
        {
            Function dense1 = TestHelper.Dense(scaledInput, hiddenLayerDim, device, Activation.Sigmoid, "");
            Function classifierOutput = TestHelper.Dense(dense1, numOutputClasses, device, Activation.None, classifierName);
            return classifierOutput;
        }

        /// <summary>
        /// Create convolution neural network
        /// </summary>
        /// <param name="features">input feature variable</param>
        /// <param name="outDims">number of output classes</param>
        /// <param name="device">CPU or GPU device to run the model</param>
        /// <param name="classifierName">name of the classifier</param>
        /// <returns>the convolution neural network classifier</returns>
        static Function CreateConvolutionalNeuralNetwork(Variable features, int outDims, DeviceDescriptor device, string classifierName)
        {
            // 28x28x1 -> 14x14x4
            int kernelWidth1 = 3, kernelHeight1 = 3, numInputChannels1 = 1, outFeatureMapCount1 = 4;
            int hStride1 = 2, vStride1 = 2;
            int poolingWindowWidth1 = 3, poolingWindowHeight1 = 3;

            Function pooling1 = ConvolutionWithMaxPooling(features, device, kernelWidth1, kernelHeight1,
                numInputChannels1, outFeatureMapCount1, hStride1, vStride1, poolingWindowWidth1, poolingWindowHeight1);

            // 14x14x4 -> 7x7x8
            int kernelWidth2 = 3, kernelHeight2 = 3, numInputChannels2 = outFeatureMapCount1, outFeatureMapCount2 = 8;
            int hStride2 = 2, vStride2 = 2;
            int poolingWindowWidth2 = 3, poolingWindowHeight2 = 3;

            Function pooling2 = ConvolutionWithMaxPooling(pooling1, device, kernelWidth2, kernelHeight2,
                numInputChannels2, outFeatureMapCount2, hStride2, vStride2, poolingWindowWidth2, poolingWindowHeight2);

            Function denseLayer = TestHelper.Dense(pooling2, outDims, device, Activation.None, classifierName);
            return denseLayer;
        }

        private static Function ConvolutionWithMaxPooling(Variable features, DeviceDescriptor device,
            int kernelWidth, int kernelHeight, int numInputChannels, int outFeatureMapCount,
            int hStride, int vStride, int poolingWindowWidth, int poolingWindowHeight)
        {
            // parameter initialization hyper parameter
            double convWScale = 0.26;
            var convParams = new Parameter(new int[] { kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount }, DataType.Float,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), device);
            Function convFunction = CNTKLib.ReLU(CNTKLib.Convolution(convParams, features, new int[] { 1, 1, numInputChannels } /* strides */));

            Function pooling = CNTKLib.Pooling(convFunction, PoolingType.Max,
                new int[] { poolingWindowWidth, poolingWindowHeight }, new int[] { hStride, vStride }, new bool[] { true });
            return pooling;
        }


        public static float ValidateModel(string modelFile, MinibatchSource testMinibatchSource,int[] imageDim, int numClasses, 
                        string featureInputName, string labelInputName, string outputName, DeviceDescriptor device, 
                        int maxCount = 1000, List<List<float>> X = null, List<float> Y = null, bool useConvolution=true)
        {
            Function model = Function.Load(modelFile, device);
            var imageInput = model.Arguments[0];
            
            var labelOutput = model.Outputs.Single(o => o.Name == outputName);

            var featureStreamInfo = testMinibatchSource.StreamInfo(featureInputName);
            var labelStreamInfo = testMinibatchSource.StreamInfo(labelInputName);

            int batchSize = 50;
            int miscountTotal = 0, totalCount = 0;

            while (true)
            {
                var minibatchData = testMinibatchSource.GetNextMinibatch((uint)batchSize, device);
                if (minibatchData == null || minibatchData.Count == 0)
                    break;
                totalCount += (int)minibatchData[featureStreamInfo].numberOfSamples;

                // expected labels are in the minibatch data.

                var labelData = minibatchData[labelStreamInfo].data.GetDenseData<float>(labelOutput);
                var expectedLabels = labelData.Select(l => l.IndexOf(l.Max())).ToList();

                var inputDataMap = new Dictionary<Variable, Value>() {
                    { imageInput, minibatchData[featureStreamInfo].data }
                };

                var outputDataMap = new Dictionary<Variable, Value>() {
                    { labelOutput, null }
                };

                model.Evaluate(inputDataMap, outputDataMap, device);

                
                var faetureData = minibatchData[featureStreamInfo].data.GetDenseData<float>(CNTKLib.InputVariable(minibatchData[featureStreamInfo].data.Shape, DataType.Float, model.Arguments[0].Name));

                var outputData = outputDataMap[labelOutput].GetDenseData<float>(labelOutput);
                var actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                miscountTotal += misMatches;
                Console.WriteLine($"Validating Model: Total Samples = {totalCount}, Misclassify Count = {miscountTotal}");

                if (totalCount > maxCount)
                {
                    //writes some result in to array 
                   
                    for (int i = 0; i < outputData.Count && X != null && Y != null; i++)
                    {
                        var imgDIm = imageDim.Aggregate(1, (acc, val) => acc * val);
                        var inputVector = faetureData[0].Skip(imgDIm * i).Take(imgDIm).Select(x => (float)x).ToList();
                        X.Add(inputVector);
                        var currLabel = actualLabels[i];
                        Y.Add(currLabel);

                    };
                    break;
                }

            }



            float errorRate = 1.0F * miscountTotal / totalCount;
            Console.WriteLine($"Model Validation Error = {errorRate}");
            return errorRate;
        }
    }
}
