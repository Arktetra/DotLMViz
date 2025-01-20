<script lang="ts">
	import OutputBlock from '../modules/OutputBlock.svelte';
	import TransformerBlocks from '../modules/TransformerBlocks.svelte';
	import UnembeddingBlock from '../modules/UnembeddingBlock.svelte';
	import EmbeddingBlock from '../modules/EmbeddingBlock.svelte';
	import InputBlock from '../modules/InputBlock.svelte';
	import TokensBlock from '../modules/TokensBlock.svelte';
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import { onMount } from 'svelte';
	import { runModel, loadModel, getAct } from './fetch.svelte';
	import { InitEventMap } from '../eventstate.svelte';
	import { active_model } from '../state.svelte';
	import ThemeToggle from '../components/ThemeToggle.svelte';

	let tokens: string[] = $state([]);
	let inpText: string = $state('');
	let activeTokenInd: number = $state(0);
	// this will hold which view it is, 0 (false) - black box view, 1 (true) - white box view
	let viewMode: boolean = $state(false);

	let act_name = 'pattern';
	let layer_name = 'attn';
	let block = 0;

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

	onMount(() => {
		InitEventMap();
		loadModel(active_model.model_name);
	});
</script>

<section
	class="flex max-h-screen min-h-[900px] min-w-[1500px] flex-col items-center justify-evenly"
>
	<div class="flex flex-row items-center justify-evenly space-x-10">
		<TokensBlock {tokens} bind:tokenInd={activeTokenInd}>
			<span class="text-sm font-light text-theme-w">
				Index: <span class="text-md font-bold">{activeTokenInd}</span>
			</span>
			<span class="text-sm font-light text-theme-w">
				Token: <span class="text-lg font-bold">'{tokens[activeTokenInd]}'</span>
			</span>
		</TokensBlock>
		<DottedBlockBase
			label="GPT-2 Small"
			titStyle="text-xl font-bold"
			borderSize={'1px'}
			inStyle="min-w-[50%] m-2 py-2 pt-8 flex-row justify-between space-x-10"
		>
			{#if viewMode}
				<EmbeddingBlock />
				<TransformerBlocks />
				<UnembeddingBlock />
			{:else}
				<div class="min-w-[40rem] min--h-[20rem] grid grid-cols-4">
					<DottedBlockBase label="Embedding"><span class="min-w-[5rem] min-h-[5rem]"></span></DottedBlockBase>
					<div class="col-span-2 relative">
						<DottedBlockBase label="Transformer Blocks" titStyle="w-full text-center text-sm"><span class="min-w-[5rem] min-h-[10rem]"></span></DottedBlockBase>
					</div>
					<DottedBlockBase label="Unembedding"><span class="min-w-[5rem] min-h-[5rem]"></span></DottedBlockBase>
				</div>
			{/if}
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
<ThemeToggle bind:state={viewMode} style="fixed bottom-5 left-[1rem] z-50" leftlabel="Detailed" rightlabel="Overview"/>
