import { NextResponse, type NextRequest } from "next/server";

export function proxy(request: NextRequest) {
  const requestHeaders = new Headers(request.headers);
  const locale = request.nextUrl.pathname === "/tr" || request.nextUrl.pathname.startsWith("/tr/")
    ? "tr"
    : "en";
  requestHeaders.set("x-app-locale", locale);

  return NextResponse.next({
    request: { headers: requestHeaders },
  });
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|icon.svg|opengraph-image|robots.txt|sitemap.xml).*)"],
};
