<script lang="ts">
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import ThemeNumberOptions from '../components/ThemeNumberOptions.svelte';
	import ElementBlockBase from '../components/ElementBlockBase.svelte';
	import AttentionHeads from './AttentionHeads.svelte';
	import Mlp from './MLP.svelte';
	import { global_state } from '../state.svelte';
	import { attnHeadCallback, LN1Callback, LN2Callback, TransformerBlockCallback } from '../callbacks.svelte';

	let pulse: boolean = $state(false);

	const _transformerBlock = [
		{
			label: 'Attention Head',
			href: '/read/attention-head'
		},
		{
			label: 'MLP',
			href: '/read/mlp'
		}
	];


	const blockChange = () => {
		pulse = true;
		TransformerBlockCallback();
		setTimeout(() => (pulse = false), 800);
	}
</script>

<DottedBlockBase label="Transformer Blocks" inStyle="flex-col p-4">
	<ThemeNumberOptions
		count={12}
		bind:activeIndex={global_state.active_block}
		clickEventCb={blockChange}
	/>
	<DottedBlockBase
		label="Block: {global_state.active_block}"
		borderSize={'1px'}
		titStyle="text-ti top-[-1.4rem]"
		inStyle="w-[30rem] h-[20rem] flex-row justify-between transition-all duration-200 {pulse ? "animate-pulse scale-75" : ""}"
	>
		<div class="flex h-full flex-col items-start justify-evenly">
			<ElementBlockBase
				blockEle={AttentionHeads}
				blockStyle="p-2 min-w-[12rem] min-h-[10rem]"
				href={_transformerBlock[0].href}
				clickEventCb={attnHeadCallback}
			>
				<span>{_transformerBlock[0].label}</span>
			</ElementBlockBase>
			<ElementBlockBase blockStyle="p-2 min-w-[4rem] min-h-[4rem]" href={'/read/layernorm'} clickEventCb={LN1Callback}>
				<span>LN</span>
			</ElementBlockBase>
		</div>
		<div class="flex h-full flex-col items-start justify-evenly">
			<ElementBlockBase blockEle={Mlp} blockStyle="p-4 min-w-[12rem] min-h-[10rem]" href={_transformerBlock[1].href}>
				<span>{_transformerBlock[1].label}</span>
			</ElementBlockBase>
			<ElementBlockBase blockStyle="p-2 min-w-[4rem] min-h-[4rem]" href={'/read/layernorm'} clickEventCb={LN2Callback}>
				<span>LN</span>
			</ElementBlockBase>
		</div>
	</DottedBlockBase>
</DottedBlockBase>
