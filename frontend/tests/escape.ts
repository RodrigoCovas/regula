const ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#x27;",
};

export function visibleMarkup(text: string): string {
  return text.replace(/[&<>"']/g, (char) => ESCAPES[char]);
}
