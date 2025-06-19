import { NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req) {
  const { question, answer } = await req.json();

  // Pure logic CoT patterns that worked well in the previous experiment
  const reasoningPatterns = [
    "Deductive Reasoning Pattern",
    "Experience-Based Reasoning Pattern", 
    "Systems Thinking Pattern",
    "Procedural/Step-by-Step Reasoning",
    "Analogical Reasoning Pattern"
  ];

  const randomPattern = reasoningPatterns[Math.floor(Math.random() * reasoningPatterns.length)];

  const prompt = `
You are a pure logic reasoning system. Generate a Chain of Thought (CoT) using ONLY abstract logical notation and mathematical/symbolic reasoning patterns. 

CRITICAL: Use NO domain-specific content. Only use:
- Variables (X, Y, Z, P1, P2, etc.)
- Logical operators (→, ∧, ∨, ¬, ∀, ∃)
- Mathematical notation (=, >, <, ≅, ∈, ⊆)
- Abstract concepts (System S, Node N, Function F, etc.)
- Pure logical relationships

Pattern to use: ${randomPattern}

Examples of PURE LOGIC CoTs:
- "Given P1: All X have property Y. Given P2: Z is an X. Therefore C: Z has property Y."
- "Pattern observed: In cases α, β, γ, solution S consistently resolved problem P. Frequency analysis shows S effectiveness = 85%."
- "System S contains nodes N1, N2, N3 with edges E1, E2, E3. Change in N1 propagates through E1 to N2."
- "Algorithm A: Step 1: Identify problem class P. Step 2: Query solution database D. Step 3: Filter solutions by constraint C."

Question: ${question}
Answer: ${answer}

Generate ONE pure logic CoT (2-3 sentences) using the ${randomPattern} pattern:`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.3, // Lower temperature for more consistent logical patterns
    max_tokens: 150,
  });

  const content = response.choices[0].message.content.trim();
  return NextResponse.json({ content });
}
