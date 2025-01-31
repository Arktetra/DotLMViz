<script lang="ts">
	import OutputBlock from '../modules/OutputBlock.svelte';
	import TransformerBlocks from '../modules/TransformerBlocks.svelte';
	import UnembeddingBlock from '../modules/UnembeddingBlock.svelte';
	import EmbeddingBlock from '../modules/EmbeddingBlock.svelte';
	import TokensBlock from '../modules/TokensBlock.svelte';
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import { onMount } from 'svelte';
	import { loadModel } from './fetch.svelte';
	import { InitEventMap } from '../eventstate.svelte';
	import { active_model, global_state, input } from '../state.svelte';
	import ExpandableDottedBlock from '../components/ExpandableDottedBlock.svelte';
	import Navbar from '../modules/Navbar.svelte';

	// let inpText: string = $state('');
	let activeTokenInd: number = $state(0);


	onMount(() => {
		InitEventMap();
		loadModel(active_model.model_name);
	});
</script>

<Navbar />
<section
	class="flex max-h-screen min-h-[900px] min-w-[1500px] flex-col items-center justify-start pt-[12rem]"
>
	<div class="flex flex-row items-start justify-evenly space-x-12 transition-all duration-300 {global_state.ouputBlockState ? "xl:pr-[25vw] pr-[20rem]" : ""}">
		<TokensBlock bind:tokenInd={activeTokenInd}>
			<span class="text-sm font-light text-theme-w">
				Index: <span class="text-md font-bold">{activeTokenInd}</span>
			</span>
			<span class="text-sm font-light text-theme-w">
				Token: <span class="text-lg font-bold">'{global_state.tokens[activeTokenInd]}'</span>
			</span>
		</TokensBlock>
		<ExpandableDottedBlock
			label="GPT-2 Small"
			href="/read/gpt2-small"
			titStyle="text-xl font-bold"
			borderSize={'1px'}
			expandCb={() => global_state.viewMode = !global_state.viewMode}
			inStyle="min-w-[50vw] h-[50vh] m-2 flex-row justify-between space-x-10"
		>
			{#if global_state.viewMode}
				<EmbeddingBlock />
				<TransformerBlocks />
				<UnembeddingBlock />
			{:else}
				<div class="min-w-[50rem] h-[40vh] grid grid-cols-4">
					<DottedBlockBase label="Embedding" inStyle="h-full py-20 flex flex-col justify-evenly items-center">
						<h1 class="tracking-tighter">Token Embedding</h1>
						<hr class="w-[80%] border-theme"/>
						<h1 class="tracking-tighter">Positional Embedding</h1>
					</DottedBlockBase>
					<div class="w-[80%] h-[95%] mx-auto col-span-2 relative mb-20">
						<DottedBlockBase label="Transformer Blocks" inStyle="h-full justify-evenly shadow-md shadow-theme relative bg-white rounded-xl" titStyle="w-full text-center text-md">
							<span class="text-3xl font-extrabold">x12</span>
							<div class="absolute w-full h-full shadow-sm shadow-theme top-3 left-3 border bg-white border-theme border-dashed rounded-xl -z-10"></div>
							<div class="absolute w-full h-full shadow-sm shadow-theme top-6 left-6 border bg-white border-theme border-dashed rounded-xl -z-20"></div>
						</DottedBlockBase>
					</div>
					<DottedBlockBase label="Unembedding" inStyle="h-full py-20 flex-col justify-evenly items-center">
						<h1 class="tracking-tighter">LayerNorm</h1>
						<hr class="w-[80%] border-theme"/>
						<h1 class="tracking-tighter">Logits</h1>
					</DottedBlockBase>
				</div>
			{/if}
		</ExpandableDottedBlock>
	</div>
</section>
<OutputBlock />
