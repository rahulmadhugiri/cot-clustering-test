import { NextResponse } from 'next/server';

export async function POST(req) {
  try {
    const { cotsData } = await req.json();
    
    if (!cotsData || !Array.isArray(cotsData)) {
      return NextResponse.json({ 
        success: false, 
        error: 'No CoTs data provided' 
      }, { status: 400 });
    }
    
    return NextResponse.json({
      success: true,
      count: cotsData.length,
      message: `Received ${cotsData.length} CoTs`,
      data: cotsData
    });
    
  } catch (error) {
    return NextResponse.json({ 
      success: false, 
      error: error.message 
    }, { status: 500 });
  }
} 