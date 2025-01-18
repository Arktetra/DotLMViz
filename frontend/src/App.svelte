<script lang="ts">
	import { onMount } from 'svelte';

	import OutputBlock from './modules/OutputBlock.svelte';
	import TransformerBlocks from './modules/TransformerBlocks.svelte';
	import UnembeddingBlock from './modules/UnembeddingBlock.svelte';
	import EmbeddingBlock from './modules/EmbeddingBlock.svelte';
	import InputBlock from './modules/InputBlock.svelte';
	import TokensBlock from './modules/TokensBlock.svelte';

	let model_name = $state('gpt2-small');
	let text = 'alpha beta gamma delta eta zeta epsilon';

	function loadModel() {
		fetch('/model/load', {
			method: 'POST',
			body: JSON.stringify({ model_name }),
			headers: {
				'Content-Type': 'application/json'
			}
		});
	}

	function runModel() {
		fetch('/model/run', {
			method: 'POST',
			body: JSON.stringify({ text }),
			headers: {
				'Content-Type': 'application/json'
			}
		});
	}

	let act_name = 'pattern';
	let layer_name = 'attn';
	let block = 0;

	async function getAttnScores() {
		const response = await fetch('/ckpt/act', {
			method: 'POST',
			body: JSON.stringify({ act_name, layer_name, block }),
			headers: {
				'Content-Type': 'application/json'
			}
		});
		var data = await response.json();
		console.log(data);
	}

	// function getAttnScores() {
	//   fetch("/ckpt/act")
	//     .then(d => d.text())
	//     .then(d => console.log(d))
	// }

	onMount(loadModel);
</script>

<!-- <button onclick={runModel}>Click here to run the model.</button>
<button onclick={getAttnScores}>Click here to get the attention scores.</button> -->

<section class="flex min-h-screen min-w-full flex-col items-center justify-evenly">
	<div class="flex min-w-[90vw] flex-row items-center justify-evenly">
		<TokensBlock />
		<EmbeddingBlock />
		<TransformerBlocks />
		<UnembeddingBlock />
	</div>

	<OutputBlock />

	<InputBlock />
</section>
