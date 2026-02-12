// =============================================================================
// TOKENIZER — converts between characters and numbers.
//
// Neural networks only understand numbers. So we assign each character an ID:
//   <BOS> = 0  (Beginning Of Sequence — signals "start of a name")
//   <EOS> = 1  (End Of Sequence — signals "this name is done")
//   'a' = 2, 'b' = 3, ... etc.
//
// "emma" becomes: [0, 6, 14, 14, 2, 1]  (BOS, e, m, m, a, EOS)
// =============================================================================

using System.Collections.Generic;
using System.Linq;

public class Tokenizer
{
    public List<string> Chars { get; }
    public Dictionary<string, int> Stoi { get; } = new(); // string-to-integer lookup
    public Dictionary<int, string> Itos { get; } = new(); // integer-to-string lookup
    public int VocabSize { get; }
    public int BOS { get; }
    public int EOS { get; }

    public Tokenizer(List<string> docs)
    {
        // Collect every unique character from all names, plus BOS and EOS markers.
        // BOS (Beginning Of Sequence): tells the model "a name starts here"
        // EOS (End Of Sequence): tells the model "this name is done"
        var allChars = new SortedSet<char>(string.Join("", docs).ToCharArray());
        Chars = new List<string> { "<BOS>", "<EOS>" };
        Chars.AddRange(allChars.Select(c => c.ToString()));
        VocabSize = Chars.Count;

        for (int i = 0; i < Chars.Count; i++)
        {
            Stoi[Chars[i]] = i;
            Itos[i] = Chars[i];
        }

        BOS = Stoi["<BOS>"];
        EOS = Stoi["<EOS>"];
    }

    /// <summary>Encode a string into a list of token IDs, with BOS/EOS markers.</summary>
    public List<int> Encode(string text)
    {
        var tokens = new List<int> { BOS };
        tokens.AddRange(text.Select(ch => Stoi[ch.ToString()]));
        tokens.Add(EOS);
        return tokens;
    }

    /// <summary>Decode a token ID back to its string representation.</summary>
    public string Decode(int tokenId) => Itos[tokenId];
}
