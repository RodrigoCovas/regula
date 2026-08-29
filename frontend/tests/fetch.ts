export type FetchCall = { url: string; init?: RequestInit };

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function recordingFetch(
  handler: (call: FetchCall) => Promise<Response> | Response,
) {
  const calls: FetchCall[] = [];
  const impl = (url: string, init?: RequestInit): Promise<Response> => {
    const call = { url, init };
    calls.push(call);
    return Promise.resolve(handler(call));
  };
  return { calls, impl };
}
