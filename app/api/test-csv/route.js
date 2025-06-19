import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const csvPath = path.join(process.cwd(), 'public', 'test.csv');
    const csvContent = fs.readFileSync(csvPath, 'utf8');
    
    const lines = csvContent.split('\n').slice(1).filter(line => line.trim());
    const parsed = lines.map((line, i) => {
      const parts = line.split(',');
      const question = parts[0];
      const answer = parts.slice(1).join(',');
      return { id: `qa-${i}`, question, answer };
    });
    
    return NextResponse.json({
      success: true,
      totalLines: csvContent.split('\n').length,
      dataRows: parsed.length,
      data: parsed
    });
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
} 