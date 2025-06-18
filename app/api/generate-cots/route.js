import { NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req) {
  const { question, answer } = await req.json();

  const prompt = `
You're an AI reasoning assistant. Given a question and an answer, generate 3 diverse chains of thought that might have led to this answer. Each chain should be 2–3 sentences long.

Question: ${question}
Answer: ${answer}

Chains of Thought:
1.
`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.7,
  });

  const content = response.choices[0].message.content;
  return NextResponse.json({ content });
}
