import { defineCollection, z } from 'astro:content';

const docs = defineCollection({
  schema: z.object({
    title: z.string(),
    description: z.string(),
    category: z
      .enum(['intro', 'idea', 'code', 'figures', 'writing', 'review', 'tools', 'library'])
      .default('tools'),
    order: z.number().default(0),
    tags: z.array(z.string()).default([]),
    pubDate: z.coerce.date().optional(),
    updatedDate: z.coerce.date().optional(),
  }),
});

export const collections = { docs };
