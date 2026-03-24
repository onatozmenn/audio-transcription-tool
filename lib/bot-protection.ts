import { checkBotId } from "botid/server";
import { NextResponse } from "next/server";

export async function denyBotTraffic(): Promise<NextResponse | null> {
  const verification = await checkBotId();
  if (verification.isBot) {
    return NextResponse.json({ error: "Access denied." }, { status: 403 });
  }

  return null;
}
