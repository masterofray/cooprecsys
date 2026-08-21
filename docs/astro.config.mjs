import { defineConfig } from 'astro/config';
import alpinejs from '@astrojs/alpinejs';
import tailwindcss from '@tailwindcss/vite';
import sitemap from '@astrojs/sitemap';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://masterofray.github.io',
  base: '/cooprecsys',
  integrations: [
    alpinejs(),
    sitemap(),
    mdx(),
  ],
  vite: {
    plugins: [tailwindcss()],
    resolve: {
      alias: {
        '@components': '/src/components',
        '../../components': '/src/components',
      },
    },
  },
  output: 'static',
});
