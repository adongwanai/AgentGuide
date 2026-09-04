export function normalizeKeyword(value: string): string {
  return value.trim().toLowerCase();
}

export function includesKeyword(text: string, keyword: string): boolean {
  const normalizedKeyword = normalizeKeyword(keyword);
  if (!normalizedKeyword) return true;
  return text.toLowerCase().includes(normalizedKeyword);
}
