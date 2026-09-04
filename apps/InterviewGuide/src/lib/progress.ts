export interface ProgressState {
  favoriteIds: string[];
  completedIds: string[];
}

export const PROGRESS_STORAGE_KEY = 'ig-question-progress-v1';

const EMPTY_PROGRESS_STATE: ProgressState = {
  favoriteIds: [],
  completedIds: [],
};

function resolveStorage(storage?: Storage | null): Storage | null {
  if (storage) return storage;
  if (typeof window === 'undefined') return null;

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function normalizeIds(input: unknown): string[] {
  if (!Array.isArray(input)) return [];
  const seen = new Set<string>();
  const ids: string[] = [];

  for (const item of input) {
    if (typeof item !== 'string') continue;
    const value = item.trim();
    if (!value || seen.has(value)) continue;
    seen.add(value);
    ids.push(value);
  }

  return ids;
}

function normalizeState(input: unknown): ProgressState {
  if (!input || typeof input !== 'object') {
    return { ...EMPTY_PROGRESS_STATE };
  }

  const raw = input as {
    favoriteIds?: unknown;
    completedIds?: unknown;
  };

  return {
    favoriteIds: normalizeIds(raw.favoriteIds),
    completedIds: normalizeIds(raw.completedIds),
  };
}

function toggleId(ids: string[], questionId: string): string[] {
  const normalizedQuestionId = questionId.trim();
  if (!normalizedQuestionId) return ids;

  if (ids.includes(normalizedQuestionId)) {
    return ids.filter((id) => id !== normalizedQuestionId);
  }

  return [...ids, normalizedQuestionId];
}

export function loadProgressState(storage?: Storage | null): ProgressState {
  const activeStorage = resolveStorage(storage);
  if (!activeStorage) return { ...EMPTY_PROGRESS_STATE };

  try {
    const raw = activeStorage.getItem(PROGRESS_STORAGE_KEY);
    if (!raw) return { ...EMPTY_PROGRESS_STATE };
    return normalizeState(JSON.parse(raw));
  } catch {
    return { ...EMPTY_PROGRESS_STATE };
  }
}

export function saveProgressState(state: ProgressState, storage?: Storage | null): ProgressState {
  const normalized = normalizeState(state);
  const activeStorage = resolveStorage(storage);
  if (!activeStorage) return normalized;

  try {
    activeStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(normalized));
  } catch {}

  return normalized;
}

export function toggleFavoriteQuestion(questionId: string, storage?: Storage | null): ProgressState {
  const current = loadProgressState(storage);
  return saveProgressState(
    {
      favoriteIds: toggleId(current.favoriteIds, questionId),
      completedIds: current.completedIds,
    },
    storage,
  );
}

export function toggleCompletedQuestion(questionId: string, storage?: Storage | null): ProgressState {
  const current = loadProgressState(storage);
  return saveProgressState(
    {
      favoriteIds: current.favoriteIds,
      completedIds: toggleId(current.completedIds, questionId),
    },
    storage,
  );
}

export function getProgressSummary(totalQuestions: number, state: ProgressState): {
  favoriteCount: number;
  completedCount: number;
  completionRate: number;
} {
  const safeTotal = Math.max(0, totalQuestions);
  const completedCount = state.completedIds.length;
  const favoriteCount = state.favoriteIds.length;
  const completionRate = safeTotal === 0 ? 0 : Math.round((completedCount / safeTotal) * 100);

  return {
    favoriteCount,
    completedCount,
    completionRate,
  };
}
