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
			},
			fontFamily: {
				main: '"Smooch Sans", serif;',
				'main-a': '"Roboto Condensed", serif'
			},
			keyframes: {
				flow: {
					'0%': {transform: 'translate(0, 0)', opacity: '0%'},
					'10%': {opacity: '30%'},
					'50%': {transform: 'translate(300%, 0)', opacity: '100%'},
					'90%': {opacity: '30%'},
					'100%': {transform: 'translate(600%, 0)', opacity: '0%'}
				}
			},
			animation: {
				'flow': 'flow 3s linear infinite'
			}
		}
	},

	plugins: []
} satisfies Config;
