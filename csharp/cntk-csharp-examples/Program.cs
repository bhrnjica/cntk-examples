using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace cntk_csharp_examples
{
    class Program
    {
        static void Main(string[] args)
        {
            CNTK.CSTrainingExamples.MNISTClassifier.TrainAndEvaluate(CNTK.DeviceDescriptor.CPUDevice, true, true);
        }
    }
}
