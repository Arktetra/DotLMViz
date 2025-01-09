/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './src/**/*.{html,js,svelte,ts}',
    './node_modules/flowbite-svelte-icons/**/*.{html,js,svelte,ts}'
  ],
  theme: {
    extend: {
      colors: {
        "theme" : "#665191",
        "theme-alt" : "#554080",
        "theme-w" : "#efefef",
        "theme-w-alt" : "#eaeaea",
      }
    },
  },
  plugins: [],
}

