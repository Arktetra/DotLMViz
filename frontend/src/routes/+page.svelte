<script lang="ts">
	import OutputBlock from '../modules/OutputBlock.svelte';
	import TransformerBlocks from '../modules/TransformerBlocks.svelte';
	import UnembeddingBlock from '../modules/UnembeddingBlock.svelte';
	import EmbeddingBlock from '../modules/EmbeddingBlock.svelte';
	import InputBlock from '../modules/InputBlock.svelte';
	import TokensBlock from '../modules/TokensBlock.svelte';
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import { onMount } from 'svelte';

	let model_name = $state('gpt2-small');
	let tokens: string[] = $state([]);
	let inpText: string = $state('');
	let activeTokenInd: number = $state(0);

	const loadModel = async () => {
		try {
			return await fetch('/model/load', {
				method: 'POST',
				body: JSON.stringify({ model_name }),
				headers: {
					'Content-Type': 'application/json'
				}
			})
				.then((res) => res)
				.catch((error) => console.log('Something not right ' + error));
		} catch (error) {
			console.log('Unable to fetch ' + error);
			return;
		}
	};

	const predNextToken = async () => {
		try {
			return await fetch('/model/pred')
				.then((res) => res)
				.then((res) => {
					let logits = res.json();
					console.log(logits);
					// return logits;
				})
				.catch((error) => console.log("Could not predict the next token" + error));
		} catch (error) {
			console.log("Unable to fetch" + error);
			return;
		}
	}

	const runModel = async () => {
		try {
			return await fetch('/model/run', {
				method: 'POST',
				body: JSON.stringify({ text: inpText }),
				headers: {
					'Content-Type': 'application/json'
				}
			})
				.then((res) => res)
				.catch((error) => console.log('Something not right ' + error));
		} catch (error) {
			console.log('Unable to fetch ' + error);
			return;
		}
	};

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
		})
			.then((res) => res)
			.catch((error) => console.log(error));
		var data = await response?.json();
		console.log(data);
	}

	// function getAttnScores() {
	//   fetch("/ckpt/act")
	//     .then(d => d.text())
	//     .then(d => console.log(d))
	// }

	const onInpChange = (v: string) => {
		inpText = v;
		genToken();
	};

	$effect(() => {
		genToken();
	});

	const genToken = () => {
		// tokens = inpText.indexOf(' ') > 0 || inpText.length > 5 ? inpText.split(' ') : inpText.split('')
		tokens = inpText.split(' ');
	};

	onMount(() => loadModel());
</script>

<section
	class="flex max-h-screen min-h-[900px] min-w-[1500px] flex-col items-center justify-evenly"
>
	<div class="flex flex-row items-center justify-evenly space-x-10">
		<TokensBlock {tokens} bind:tokenInd={activeTokenInd}>
			<span class="text-sm font-light text-theme-w">Index: <span class="text-md font-bold">{activeTokenInd}</span></span>
			<span class="text-sm font-light text-theme-w">Token: <span class="text-lg font-bold">'{tokens[activeTokenInd]}'</span></span>
		</TokensBlock>
		<DottedBlockBase
			label="GPT-2 Small"
			titStyle="text-xl font-bold"
			borderSize={'1px'}
			inStyle="min-w-[50%] m-2 py-2 pt-8 flex-row justify-between space-x-10"
		>
			<EmbeddingBlock />
			<TransformerBlocks />
			<UnembeddingBlock />
		</DottedBlockBase>
		<DottedBlockBase label="Output">
			<div class="flex min-h-[5rem] min-w-[5rem] flex-col items-center justify-evenly">
				<span class="rounded-md bg-theme p-1 px-2 font-light text-theme-w">E</span>
			</div>
		</DottedBlockBase>
	</div>

	<InputBlock bind:value={inpText} inpEventCb={onInpChange} />
</section>
<OutputBlock />
