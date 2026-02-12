// =============================================================================
// NEURAL OPS — stateless building blocks used by the GPT model.
// =============================================================================

using System.Linq;

public static class NeuralOps
{
    // -------------------------------------------------------------------------
    // Linear: Matrix-vector multiplication. The fundamental operation of neural nets.
    //
    // Takes an input vector x and a weight matrix w, returns w * x.
    // Each output element is a weighted sum of all inputs — the weights decide
    // which inputs matter and how much.
    //
    // Example: if x = [0.5, 0.3] and w = [[2, 1], [0, 3]], then:
    //   output[0] = 2*0.5 + 1*0.3 = 1.3
    //   output[1] = 0*0.5 + 3*0.3 = 0.9
    //
    // The magic: these weights are learned during training. The model discovers
    // which combinations of input features are useful for prediction.
    // -------------------------------------------------------------------------
    public static Value[] Linear(Value[] x, Value[][] w)
    {
        var result = new Value[w.Length];
        for (int o = 0; o < w.Length; o++)
        {
            Value sum = new Value(0);
            for (int i = 0; i < x.Length; i++)
                sum = sum + w[o][i] * x[i];
            result[o] = sum;
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Softmax: Converts raw scores into probabilities that sum to 1.
    //
    // Input:  [2.0, 1.0, 0.1]  (raw scores, called "logits")
    // Output: [0.66, 0.24, 0.10]  (probabilities)
    //
    // Higher scores get higher probabilities. The exponential makes big
    // differences more extreme — if one score is much higher, it dominates.
    //
    // The "subtract max" trick prevents numerical overflow. e^1000 is infinity,
    // but e^0 is fine. Subtracting the max doesn't change the relative
    // probabilities (it cancels out in the division).
    // -------------------------------------------------------------------------
    public static Value[] Softmax(Value[] logits)
    {
        double maxVal = logits.Max(v => v.Data);
        var exps = logits.Select(v => (v - maxVal).Exp()).ToArray();
        Value total = exps.Aggregate(new Value(0), (acc, e) => acc + e);
        return exps.Select(e => e / total).ToArray();
    }

    // -------------------------------------------------------------------------
    // RMSNorm (Root Mean Square Normalization): Keeps numbers in a healthy range.
    //
    // Without normalization, values can grow or shrink as they pass through layers,
    // making training unstable. RMSNorm scales the vector so its average squared
    // magnitude is ~1. Think of it like auto-adjusting the volume.
    //
    // This is a simpler alternative to LayerNorm (used in the original GPT-2).
    // LLaMA and most modern models use RMSNorm because it works just as well
    // with fewer operations.
    // -------------------------------------------------------------------------
    public static Value[] RMSNorm(Value[] x)
    {
        Value ms = new Value(0);
        foreach (var xi in x)
            ms = ms + xi * xi;
        ms = ms / x.Length;
        var scale = (ms + 1e-5).Pow(-0.5); // 1e-5 prevents division by zero
        return x.Select(xi => xi * scale).ToArray();
    }
}
