import { describe, expect, it } from 'vitest';
import {
  PROGRESS_STORAGE_KEY,
  loadProgressState,
  saveProgressState,
  toggleCompletedQuestion,
  toggleFavoriteQuestion,
} from '../../src/lib/progress';

function createMemoryStorage(): Storage {
  const map = new Map<string, string>();

  return {
    get length() {
      return map.size;
    },
    clear() {
      map.clear();
    },
    getItem(key: string) {
      return map.has(key) ? map.get(key)! : null;
    },
    key(index: number) {
      return Array.from(map.keys())[index] ?? null;
    },
    removeItem(key: string) {
      map.delete(key);
    },
    setItem(key: string, value: string) {
      map.set(key, value);
    },
  };
}

describe('progress storage wrapper', () => {
  it('returns empty state when storage is empty', () => {
    const storage = createMemoryStorage();
    expect(loadProgressState(storage)).toEqual({
      favoriteIds: [],
      completedIds: [],
    });
  });

  it('toggles favorite/completed and persists to storage', () => {
    const storage = createMemoryStorage();

    toggleFavoriteQuestion('q-1', storage);
    toggleCompletedQuestion('q-2', storage);
    toggleCompletedQuestion('q-3', storage);

    expect(loadProgressState(storage)).toEqual({
      favoriteIds: ['q-1'],
      completedIds: ['q-2', 'q-3'],
    });

    toggleFavoriteQuestion('q-1', storage);
    toggleCompletedQuestion('q-2', storage);

    expect(loadProgressState(storage)).toEqual({
      favoriteIds: [],
      completedIds: ['q-3'],
    });
  });

  it('falls back to empty state on invalid json', () => {
    const storage = createMemoryStorage();
    storage.setItem(PROGRESS_STORAGE_KEY, '{ bad json ');

    expect(loadProgressState(storage)).toEqual({
      favoriteIds: [],
      completedIds: [],
    });
  });

  it('deduplicates and sanitizes ids before save', () => {
    const storage = createMemoryStorage();

    saveProgressState(
      {
        favoriteIds: ['q-1', '', 'q-1', '  ', 'q-2'],
        completedIds: ['q-3', 'q-3', 'q-4'],
      },
      storage,
    );

    expect(loadProgressState(storage)).toEqual({
      favoriteIds: ['q-1', 'q-2'],
      completedIds: ['q-3', 'q-4'],
    });
  });
});
