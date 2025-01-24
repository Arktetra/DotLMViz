import type { Config } from 'tailwindcss';

export default {
	content: [
		'./src/**/*.{html,js,svelte,ts}',
		'./node_modules/flowbite-svelte-icons/**/*.{html,js,svelte,ts}'
	],
	theme: {
		extend: {
			colors: {
				theme: '#665191',
				'theme-alt': '#554080',
				'theme-w': '#efefef',
				'theme-w-alt': '#eaeaea',
				'theme-g': '#dddddd',
				'theme-g-alt': '#cdcdcd',
				'theme-r': '#ec2315',
				'theme-r-alt': '#db1204'
			},
			fontSize: {
				ti: '0.75rem',
				'ti-s': '0.6rem'
			}
		}
	},

	plugins: []
} satisfies Config;
