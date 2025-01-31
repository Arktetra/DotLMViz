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

	// let inpText: string = $state('');
	let activeTokenInd: number = $state(0);


	onMount(() => {
		InitEventMap();
		loadModel(active_model.model_name);
	});
</script>

<section
	class="flex max-h-screen min-h-[900px] min-w-[1500px] flex-col items-center justify-start pt-[12rem]"
>
	<div class="flex flex-row items-start justify-evenly space-x-12 transition-all duration-300 {global_state.ouputBlockState ? "pr-[25rem]" : ""}">
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
			inStyle="min-w-[50%] m-2 flex-row justify-between space-x-10"
		>
			{#if global_state.viewMode}
				<EmbeddingBlock />
				<TransformerBlocks />
				<UnembeddingBlock />
			{:else}
				<div class="min-w-[40rem] min-h-[20rem] grid grid-cols-4">
					<DottedBlockBase label="Embedding"><span class="min-w-[5rem] min-h-[5rem]"></span></DottedBlockBase>
					<div class="col-span-2 relative">
						<DottedBlockBase label="Transformer Blocks" titStyle="w-full text-center text-sm"><span class="min-w-[5rem] min-h-[20rem]"></span></DottedBlockBase>
					</div>
					<DottedBlockBase label="Unembedding"><span class="min-w-[5rem] min-h-[5rem]"></span></DottedBlockBase>
				</div>
			{/if}
		</ExpandableDottedBlock>
	</div>
</section>
<OutputBlock />
