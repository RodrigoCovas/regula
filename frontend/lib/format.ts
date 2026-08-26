export function joinLabel(
  ...parts: Array<string | null | undefined>
): string {
  return parts.filter(Boolean).join(" — ");
}
