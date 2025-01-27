<script lang="ts">
	import OutputBlock from '../modules/OutputBlock.svelte';
	import TransformerBlocks from '../modules/TransformerBlocks.svelte';
	import UnembeddingBlock from '../modules/UnembeddingBlock.svelte';
	import EmbeddingBlock from '../modules/EmbeddingBlock.svelte';
	import InputBlock from '../modules/InputBlock.svelte';
	import TokensBlock from '../modules/TokensBlock.svelte';
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import { onMount } from 'svelte';
	import { runModel, loadModel, getAct, getTokens } from './fetch.svelte';
	import { InitEventMap } from '../eventstate.svelte';
	import { active_model, global_state, input } from '../state.svelte';
	import ThemeToggle from '../components/ThemeToggle.svelte';
	import ExpandableDottedBlock from '../components/ExpandableDottedBlock.svelte';

	// let inpText: string = $state('');
	let activeTokenInd: number = $state(0);
	// this will hold which view it is, 0 (false) - black box view, 1 (true) - white box view
	let viewMode: boolean = $state(false);

	const onInpChange = (v: string) => {
		input.text = v;
		genToken();
		input.isChanged = true;
	};

	$effect(() => {
		genToken();
	});

	const genToken = async () => {
		await getTokens(input.text);
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
		<TokensBlock tokens={global_state.tokens} bind:tokenInd={activeTokenInd}>
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
			expandCb={() => viewMode = !viewMode}
			inStyle="min-w-[50%] m-2 flex-row justify-between space-x-10"
		>
			{#if viewMode}
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
		<DottedBlockBase label="Output">
			<div class="flex min-h-[5rem] min-w-[5rem] flex-col items-center justify-evenly">
				<span class="rounded-md bg-theme p-1 px-2 font-light text-theme-w">E</span>
			</div>
		</DottedBlockBase>
	</div>

	<InputBlock bind:value={input.text} inpEventCb={onInpChange} />
</section>
<OutputBlock />
<ThemeToggle bind:state={viewMode} style="fixed bottom-5 left-[1rem] z-50" leftlabel="Detailed" rightlabel="Overview"/>
