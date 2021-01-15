using Logic;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace GUI
{
    public partial class ReactionDiffusionGUI : Window
    {
        private IReactionDiffusion ReactionDiffusion;
        private float DiffusionRateA;
        private float DiffusionRateB;
        private float FeedRate;
        private float KillRate;

        private readonly DispatcherTimer Timer;
        private readonly Stopwatch Stopwatch;
        private List<int> FrameCalculationDurations;

        private enum OperationMode { WithoutOpenCL, WithOpenCL, WithOpenCLAndLocalMemory }

        public ReactionDiffusionGUI()
        {
            InitializeComponent();         

            Timer = new DispatcherTimer();
            Timer.Interval = new TimeSpan(166_666); // 16 666 666,66.. ns 16 666,66.. µs 16,66.. ms 60fps
            Timer.Tick += new EventHandler(CalculateNextFrame);

            Stopwatch = new Stopwatch();
            FrameCalculationDurations = new List<int>();      
        }

        private void ButtonStartDiffusion_Click(object sender, RoutedEventArgs e)
        {
            StartDiffusion(OperationMode.WithoutOpenCL);
        }

        private void ButtonStartDiffusionOpenCL_Click(object sender, RoutedEventArgs e)
        {
            StartDiffusion(OperationMode.WithOpenCL);
        }

        private void ButtonStartDiffusionOpenCLWithLocalMem_Click(object sender, RoutedEventArgs e)
        {
            StartDiffusion(OperationMode.WithOpenCLAndLocalMemory);
        }

        private void StartDiffusion(OperationMode operationMode)
        {
            if (ValidateInputParameters())
            {
                if (operationMode == OperationMode.WithoutOpenCL)
                {
                    CreateDiffuser();
                }
                else if (operationMode == OperationMode.WithOpenCL)
                {
                    CreateDiffuserWithOpenCL();
                }
                else
                {
                    CreateDiffuserWithOpenCLAndLocalMemory();
                }
                FrameCalculationDurations = new List<int>();
                Timer.Start();
            }
            else
            {
                MessageBox.Show("Faulty parameters");
            }
        }

        // Multiculture parsing: https://stackoverflow.com/questions/1354924/how-do-i-parse-a-string-with-a-decimal-point-to-a-double
        private bool ValidateInputParameters()
        {
            return (float.TryParse(TextBoxDiffRateA.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out DiffusionRateA) &&
                    float.TryParse(TextBoxDiffRateB.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out DiffusionRateB) &&
                    float.TryParse(TextBoxFeedRate.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out FeedRate) &&
                    float.TryParse(TextBoxKillRate.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out KillRate));
        }

        private void CreateDiffuser()
        {
            ReactionDiffusion = new ReactionDiffusion((int)Canvas.ActualWidth, (int)Canvas.ActualHeight, DiffusionRateA, DiffusionRateB, FeedRate, KillRate);
        }

        private void CreateDiffuserWithOpenCL()
        {
            ReactionDiffusion = new ReactionDiffusionOpenCL((int)Canvas.ActualWidth, (int)Canvas.ActualHeight, DiffusionRateA, DiffusionRateB, FeedRate, KillRate, false);
        }

        private void CreateDiffuserWithOpenCLAndLocalMemory()
        {
            ReactionDiffusion = new ReactionDiffusionOpenCL((int)Canvas.ActualWidth, (int)Canvas.ActualHeight, DiffusionRateA, DiffusionRateB, FeedRate, KillRate, true);
        }

        private void CalculateNextFrame(Object sender, EventArgs e)
        {
            InvalidateVisual();
        }

        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);

            if (Timer.IsEnabled)
            {
                Stopwatch.Start();
                BitmapSource diffuseFrame = ReactionDiffusion.Diffuse();                                         
                Stopwatch.Stop();

                AddBitmapSourceToDrawingContext(drawingContext, diffuseFrame);
                DisplayGeneratedFrameOnScreen(diffuseFrame);
                UpdateCalculationLabels();
            }
        }

        private void AddBitmapSourceToDrawingContext(DrawingContext drawingContext, BitmapSource diffuseFrame)
        {
            Rect rect = new Rect(0, 0, (int)Canvas.ActualWidth, (int)Canvas.ActualHeight);

            using (drawingContext = new DrawingVisual().RenderOpen())
            {
                drawingContext.DrawImage(diffuseFrame, rect);
            }
        }

        private void DisplayGeneratedFrameOnScreen(BitmapSource diffuseFrame)
        {
            Image image = new Image();
            image.Source = diffuseFrame;
            Canvas.Children.Clear();
            Canvas.Children.Add(image);
        }

        private void UpdateCalculationLabels()
        {
            FrameCalculationDurations.Add((int)Stopwatch.ElapsedTicks);
            Stopwatch.Reset();
            LabelCalcFrameDurationFps.Content = "Frames per sec [fps]: " + 1 / (FrameCalculationDurations.Average() / 10_000_000);
            LabelCalcFrameDuration.Content = "Calc time [ticks]: " + Math.Round(FrameCalculationDurations.Average());
        }
    }
}
