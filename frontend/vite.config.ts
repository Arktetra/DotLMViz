import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/ckpt/act': 'http://127.0.0.1:5000',
			'/dimred/pca': 'http://127.0.0.1:5000',
			'/model/load': {
				target: 'http://127.0.0.1:5000',
				timeout: 10000	// don't abandon the connection until 10 seconds have passed, sometimes loading model takes around 5 seconds.
			},
			'/model/pred': 'http://127.0.0.1:5000',
			'/model/run': 'http://127.0.0.1:5000',
		}
	}
});
