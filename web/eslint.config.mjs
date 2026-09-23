// ESLint 9 flat config wrapping Next's shareable configs.
// Replaces `next lint`, which is deprecated in Next 15.5 and removed in 16.
import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __dirname = dirname(fileURLToPath(import.meta.url));
const compat = new FlatCompat({ baseDirectory: __dirname });

const eslintConfig = [
  { ignores: [".next/**", "out/**", "node_modules/**", "next-env.d.ts"] },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // Pre-existing: the Masthead wordmark is a plain <a href="/">, which does
      // a full reload (and so resets in-page state) rather than a client-side
      // transition. Kept as a warning, not an error, until someone decides
      // whether that reload is intended.
      "@next/next/no-html-link-for-pages": "warn",
    },
  },
];

export default eslintConfig;
