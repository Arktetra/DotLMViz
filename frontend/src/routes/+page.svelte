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
	import Message from '../components/Message.svelte';

	// let inpText: string = $state('');
	let activeTokenInd: number = $state(0);

	onMount(() => {
		InitEventMap();
		loadModel(active_model.model_name);
	});

	let showRead = $state(true)

</script>

{#if showRead}
<Message 
	type="info" 
	ostyle="bg-theme bottom-10 right-[50%] translate-x-[50%] !text-sm"
	message="Recomendation: Please visit reading section first to understand the basic theories" 
>
	<a href={'/read'} class="hover:underline ml-5">Visit Now</a>
	<button onclick={() => showRead = false} class="ml-5 hover:scale-125">X</button>
</Message>
{/if}

<Navbar />
<section
	class="flex max-h-screen min-h-[900px] min-w-[1500px] flex-col items-center justify-start pt-[10rem] font-main-a"
>
	<div
		class="flex w-full flex-row items-start justify-evenly space-x-6 transition-all duration-300 xl:min-w-[70%] {global_state.ouputBlockState
			? 'pr-0 xl:pr-[27rem]'
			: ''}"
	>
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
			titStyle="text-xl font-bold text-black"
			borderSize={'1px'}
			expandCb={() => (global_state.viewMode = !global_state.viewMode)}
			inStyle="min-w-[60rem] h-[32rem] m-3 flex-row justify-between space-x-5"
		>
			{#if global_state.viewMode}
				<EmbeddingBlock />
				<TransformerBlocks />
				<UnembeddingBlock />
			{:else}
				<div class="grid h-[27rem] w-full grid-cols-4">
					<DottedBlockBase
						label="Embedding"
						inStyle="h-full py-20 flex flex-col justify-evenly items-center"
					>
						<h1 class="tracking-tighter">Token Embedding</h1>
						<hr class="w-[80%] border-theme" />
						<h1 class="tracking-tighter">Positional Embedding</h1>
					</DottedBlockBase>
					<div class="relative col-span-2 mx-auto mb-20 h-[95%] w-[80%]">
						<DottedBlockBase
							label="Transformer Blocks"
							inStyle="h-full justify-evenly shadow-md shadow-theme relative bg-white rounded-xl"
							titStyle="w-full text-center text-md"
						>
							<span class="text-3xl font-extrabold">x12</span>
							<div
								class="absolute left-3 top-3 -z-10 h-full w-full rounded-xl border border-dashed border-theme bg-white shadow-sm shadow-theme"
							></div>
							<div
								class="absolute left-6 top-6 -z-20 h-full w-full rounded-xl border border-dashed border-theme bg-white shadow-sm shadow-theme"
							></div>
						</DottedBlockBase>
					</div>
					<div class="grid h-full grid-rows-2 gap-10">
						<DottedBlockBase
							label="LayerNorm"
							inStyle="h-full py-20 flex-col justify-evenly items-center"
						></DottedBlockBase>
						<DottedBlockBase
							label="Unembedding"
							inStyle="h-full py-20 flex-col justify-evenly items-center"
						></DottedBlockBase>
					</div>
				</div>
			{/if}
		</ExpandableDottedBlock>
	</div>
</section>
<OutputBlock />
