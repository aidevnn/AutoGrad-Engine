// =============================================================================
// AUTOGRAD ENGINE TESTS
// =============================================================================
//
// These tests verify that the Value class correctly computes both forward
// results and backward gradients. The key technique is "numerical gradient
// checking": for each operation, we compare the analytically computed gradient
// (from backprop) against a numerically estimated gradient (by nudging inputs
// by a tiny epsilon and measuring the output change).
//
// If the analytic and numeric gradients match (within a tolerance), the
// backward pass is correct. This is the standard way to verify autograd
// implementations — PyTorch uses the same approach in torch.autograd.gradcheck.
// =============================================================================

using Xunit;

namespace AutogradEngine.Tests;

public class ValueTests
{
    private const double Epsilon = 1e-6;
    private const double GradTolerance = 1e-4;

    /// <summary>
    /// Numerically estimates the gradient of a function with respect to a Value.
    /// Uses the central difference method: (f(x+h) - f(x-h)) / (2h).
    /// This is more accurate than the one-sided (f(x+h) - f(x)) / h approach.
    /// </summary>
    private static double NumericalGradient(Func<double, double> f, double x)
    {
        return (f(x + Epsilon) - f(x - Epsilon)) / (2 * Epsilon);
    }

    // =========================================================================
    // BASIC OPERATIONS — verify forward results and gradient flow
    // =========================================================================

    [Fact]
    public void Addition_ForwardAndBackward()
    {
        var a = new Value(2.0);
        var b = new Value(3.0);
        var c = a + b;
        c.Backward();

        Assert.Equal(5.0, c.Data, precision: 10);
        Assert.Equal(1.0, a.Grad, precision: 10); // d(a+b)/da = 1
        Assert.Equal(1.0, b.Grad, precision: 10); // d(a+b)/db = 1
    }

    [Fact]
    public void Multiplication_ForwardAndBackward()
    {
        var a = new Value(2.0);
        var b = new Value(3.0);
        var c = a * b;
        c.Backward();

        Assert.Equal(6.0, c.Data, precision: 10);
        Assert.Equal(3.0, a.Grad, precision: 10); // d(a*b)/da = b = 3
        Assert.Equal(2.0, b.Grad, precision: 10); // d(a*b)/db = a = 2
    }

    [Fact]
    public void Subtraction_ForwardAndBackward()
    {
        var a = new Value(5.0);
        var b = new Value(3.0);
        var c = a - b;
        c.Backward();

        Assert.Equal(2.0, c.Data, precision: 10);
        Assert.Equal(1.0, a.Grad, precision: 10);  // d(a-b)/da =  1
        Assert.Equal(-1.0, b.Grad, precision: 10); // d(a-b)/db = -1
    }

    [Fact]
    public void Division_ForwardAndBackward()
    {
        var a = new Value(6.0);
        var b = new Value(3.0);
        var c = a / b;
        c.Backward();

        Assert.Equal(2.0, c.Data, precision: 5);

        // Numerical gradient check
        double numGradA = NumericalGradient(x => x / 3.0, 6.0);
        double numGradB = NumericalGradient(x => 6.0 / x, 3.0);
        Assert.Equal(numGradA, a.Grad, GradTolerance);
        Assert.Equal(numGradB, b.Grad, GradTolerance);
    }

    [Fact]
    public void Negation_ForwardAndBackward()
    {
        var a = new Value(4.0);
        var c = -a;
        c.Backward();

        Assert.Equal(-4.0, c.Data, precision: 10);
        Assert.Equal(-1.0, a.Grad, precision: 10);
    }

    [Fact]
    public void Power_ForwardAndBackward()
    {
        var a = new Value(3.0);
        var c = a.Pow(2);
        c.Backward();

        Assert.Equal(9.0, c.Data, precision: 10);
        Assert.Equal(6.0, a.Grad, precision: 10); // d(x^2)/dx = 2x = 6

        // Numerical gradient check
        double numGrad = NumericalGradient(x => Math.Pow(x, 2), 3.0);
        Assert.Equal(numGrad, a.Grad, GradTolerance);
    }

    [Fact]
    public void Exp_ForwardAndBackward()
    {
        var a = new Value(1.0);
        var c = a.Exp();
        c.Backward();

        Assert.Equal(Math.E, c.Data, precision: 10);
        Assert.Equal(Math.E, a.Grad, precision: 10); // d(e^x)/dx = e^x

        // Numerical gradient check
        double numGrad = NumericalGradient(Math.Exp, 1.0);
        Assert.Equal(numGrad, a.Grad, GradTolerance);
    }

    [Fact]
    public void Log_ForwardAndBackward()
    {
        var a = new Value(2.0);
        var c = a.Log();
        c.Backward();

        Assert.Equal(Math.Log(2.0), c.Data, precision: 10);
        Assert.Equal(0.5, a.Grad, precision: 10); // d(ln(x))/dx = 1/x = 0.5

        // Numerical gradient check
        double numGrad = NumericalGradient(Math.Log, 2.0);
        Assert.Equal(numGrad, a.Grad, GradTolerance);
    }

    [Fact]
    public void ReLU_Positive_PassesThrough()
    {
        var a = new Value(3.0);
        var c = a.ReLU();
        c.Backward();

        Assert.Equal(3.0, c.Data, precision: 10);
        Assert.Equal(1.0, a.Grad, precision: 10); // x > 0 → gradient = 1
    }

    [Fact]
    public void ReLU_Negative_OutputsZero()
    {
        var a = new Value(-3.0);
        var c = a.ReLU();
        c.Backward();

        Assert.Equal(0.0, c.Data, precision: 10);
        Assert.Equal(0.0, a.Grad, precision: 10); // x < 0 → gradient = 0
    }

    // =========================================================================
    // COMPOSITE EXPRESSIONS — verify chain rule through multi-step computations
    // =========================================================================

    [Fact]
    public void ChainedOperations_CorrectGradients()
    {
        // f = (a * b) + c  where a=2, b=3, c=5
        // f = 6 + 5 = 11
        // df/da = b = 3, df/db = a = 2, df/dc = 1
        var a = new Value(2.0);
        var b = new Value(3.0);
        var c = new Value(5.0);
        var f = (a * b) + c;
        f.Backward();

        Assert.Equal(11.0, f.Data, precision: 10);
        Assert.Equal(3.0, a.Grad, precision: 10);
        Assert.Equal(2.0, b.Grad, precision: 10);
        Assert.Equal(1.0, c.Grad, precision: 10);
    }

    [Fact]
    public void SquaredReLU_GradientCheck()
    {
        // This is the activation function used in the MLP: ReLU(x)^2
        var a = new Value(2.0);
        var c = a.ReLU().Pow(2);
        c.Backward();

        Assert.Equal(4.0, c.Data, precision: 10);

        // Numerical gradient check for the composed function
        double numGrad = NumericalGradient(x => Math.Pow(Math.Max(0, x), 2), 2.0);
        Assert.Equal(numGrad, a.Grad, GradTolerance);
    }

    [Fact]
    public void SameVariableUsedTwice_GradientsAccumulate()
    {
        // f = a + a = 2a → df/da = 2
        var a = new Value(3.0);
        var c = a + a;
        c.Backward();

        Assert.Equal(6.0, c.Data, precision: 10);
        Assert.Equal(2.0, a.Grad, precision: 10);
    }

    [Fact]
    public void SameVariableMultipliedBySelf_GradientsCorrect()
    {
        // f = a * a = a^2 → df/da = 2a = 6
        var a = new Value(3.0);
        var c = a * a;
        c.Backward();

        Assert.Equal(9.0, c.Data, precision: 10);
        Assert.Equal(6.0, a.Grad, precision: 10);
    }

    [Fact]
    public void ComplexExpression_NumericalGradientCheck()
    {
        // f = (a * b + c).Exp().Log() — chain of non-linear operations
        // This should round-trip: log(exp(x)) = x, but gradients must propagate correctly
        double aVal = 1.5, bVal = 2.0, cVal = -1.0;

        var a = new Value(aVal);
        var b = new Value(bVal);
        var c = new Value(cVal);
        var f = (a * b + c).Exp().Log();
        f.Backward();

        // Numerical gradient checks
        double numGradA = NumericalGradient(x => Math.Log(Math.Exp(x * bVal + cVal)), aVal);
        double numGradB = NumericalGradient(x => Math.Log(Math.Exp(aVal * x + cVal)), bVal);
        double numGradC = NumericalGradient(x => Math.Log(Math.Exp(aVal * bVal + x)), cVal);

        Assert.Equal(numGradA, a.Grad, GradTolerance);
        Assert.Equal(numGradB, b.Grad, GradTolerance);
        Assert.Equal(numGradC, c.Grad, GradTolerance);
    }

    // =========================================================================
    // NEURAL NETWORK BUILDING BLOCKS — verify the ops used in GPT
    // =========================================================================

    [Fact]
    public void Softmax_SumsToOne()
    {
        var logits = new[] { new Value(2.0), new Value(1.0), new Value(0.1) };
        var probs = NeuralOps.Softmax(logits);

        double sum = probs.Sum(p => p.Data);
        Assert.Equal(1.0, sum, precision: 10);
    }

    [Fact]
    public void Softmax_HighestLogitGetsHighestProbability()
    {
        var logits = new[] { new Value(5.0), new Value(1.0), new Value(0.1) };
        var probs = NeuralOps.Softmax(logits);

        Assert.True(probs[0].Data > probs[1].Data);
        Assert.True(probs[1].Data > probs[2].Data);
    }

    [Fact]
    public void Softmax_UniformInputs_UniformOutput()
    {
        var logits = new[] { new Value(1.0), new Value(1.0), new Value(1.0) };
        var probs = NeuralOps.Softmax(logits);

        foreach (var p in probs)
            Assert.Equal(1.0 / 3.0, p.Data, precision: 10);
    }

    [Fact]
    public void Softmax_GradientFlowsBack()
    {
        var logits = new[] { new Value(2.0), new Value(1.0), new Value(0.5) };
        var probs = NeuralOps.Softmax(logits);

        // Take -log of first probability (cross-entropy for target=0)
        var loss = -(probs[0].Log());
        loss.Backward();

        // All logits should have non-zero gradients
        Assert.NotEqual(0.0, logits[0].Grad);
        Assert.NotEqual(0.0, logits[1].Grad);
        Assert.NotEqual(0.0, logits[2].Grad);
    }

    [Fact]
    public void Linear_ForwardCorrectness()
    {
        // Simple 2x2 linear: y = W * x
        var x = new[] { new Value(1.0), new Value(2.0) };
        var w = new[]
        {
            new[] { new Value(3.0), new Value(4.0) },  // row 0: 3*1 + 4*2 = 11
            new[] { new Value(5.0), new Value(6.0) }   // row 1: 5*1 + 6*2 = 17
        };
        var y = NeuralOps.Linear(x, w);

        Assert.Equal(11.0, y[0].Data, precision: 10);
        Assert.Equal(17.0, y[1].Data, precision: 10);
    }

    [Fact]
    public void Linear_GradientFlowsToWeightsAndInputs()
    {
        var x = new[] { new Value(1.0), new Value(2.0) };
        var w = new[]
        {
            new[] { new Value(3.0), new Value(4.0) },
            new[] { new Value(5.0), new Value(6.0) }
        };
        var y = NeuralOps.Linear(x, w);

        // Sum outputs to get a scalar loss, then backprop
        var loss = y[0] + y[1];
        loss.Backward();

        // Both inputs and all weights should have gradients
        Assert.NotEqual(0.0, x[0].Grad);
        Assert.NotEqual(0.0, x[1].Grad);
        Assert.NotEqual(0.0, w[0][0].Grad);
        Assert.NotEqual(0.0, w[1][1].Grad);
    }

    [Fact]
    public void RMSNorm_NormalizesVector()
    {
        var x = new[] { new Value(3.0), new Value(4.0) };
        var normed = NeuralOps.RMSNorm(x);

        // After RMSNorm, the RMS of the output should be approximately 1
        double rms = Math.Sqrt(normed.Sum(v => v.Data * v.Data) / normed.Length);
        Assert.Equal(1.0, rms, precision: 3);
    }

    [Fact]
    public void RMSNorm_GradientFlowsBack()
    {
        var x = new[] { new Value(3.0), new Value(4.0) };
        var normed = NeuralOps.RMSNorm(x);
        var loss = normed[0] + normed[1];
        loss.Backward();

        Assert.NotEqual(0.0, x[0].Grad);
        Assert.NotEqual(0.0, x[1].Grad);
    }

    // =========================================================================
    // TOKENIZER — verify encode/decode roundtrip
    // =========================================================================

    [Fact]
    public void Tokenizer_EncodeDecode_Roundtrips()
    {
        var tokenizer = new Tokenizer(new List<string> { "emma", "olivia" });
        var tokens = tokenizer.Encode("emma");

        Assert.Equal(tokenizer.BOS, tokens[0]);
        Assert.Equal(tokenizer.EOS, tokens[^1]);

        // Decode middle tokens back to characters
        var decoded = string.Join("", tokens.Skip(1).Take(tokens.Count - 2)
            .Select(t => tokenizer.Decode(t)));
        Assert.Equal("emma", decoded);
    }

    [Fact]
    public void Tokenizer_VocabSizeIncludesBosAndEos()
    {
        var tokenizer = new Tokenizer(new List<string> { "abc" });
        // 'a', 'b', 'c' + BOS + EOS = 5
        Assert.Equal(5, tokenizer.VocabSize);
    }
}
