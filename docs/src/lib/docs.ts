import { getCollection, type CollectionEntry } from 'astro:content';
import { BASE_PATH } from './site-config.ts';

export async function getAllDocs() {
  const docs = await getCollection('docs');
  return docs.sort((a: CollectionEntry<'docs'>, b: CollectionEntry<'docs'>) => {
    if (a.data.order !== b.data.order) return a.data.order - b.data.order;
    return a.data.title.localeCompare(b.data.title);
  });
}

export async function getDocsByCategory() {
  const allDocs = await getAllDocs();
  const categories: Record<string, CollectionEntry<'docs'>[]> = {};
  for (const doc of allDocs) {
    const category = doc.data.category || 'Uncategorized';
    (categories[category] ||= []).push(doc);
  }
  return categories;
}

export async function getCategories() {
  const docs = await getCollection('docs');
  return Array.from(
    new Set(docs.map((doc) => doc.data.category).filter(Boolean)),
  ).sort();
}

export async function getDocsNav() {
  const categories = await getDocsByCategory();
  return Object.keys(categories).sort().map((category) => ({
    name: category,
    items: categories[category],
  }));
}

export function getDocSlug(doc: Pick<CollectionEntry<'docs'>, 'id'>): string {
  return doc.id.replace(/\.(md|mdx)$/i, '');
}

export function getDocUrl(doc: Pick<CollectionEntry<'docs'>, 'id'>): string {
  return `${BASE_PATH}/docs/${getDocSlug(doc)}`;
}

export function getCategoryUrl(category: string): string {
  return `${BASE_PATH}/docs/category/${category.toLowerCase().replace(/\s+/g, '-')}`;
}
