<script lang="ts">
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import ThemeNumberOptions from '../components/ThemeNumberOptions.svelte';
	import ElementBlockBase from '../components/ElementBlockBase.svelte';
	import AttentionHeads from './AttentionHeads.svelte';
	import Mlp from './MLP.svelte';
	import { activeComponent, global_state, input } from '../state.svelte';
	import { getAttnPattern, getProbDensity, runModel } from '../routes/fetch.svelte';
	import { active } from 'd3';

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

	const LN1Callback = async () => {
		if (input.isChanged === true) {
			await runModel(input.text);
		}

		await getProbDensity("resid_pre", null, global_state.active_block);
		await getProbDensity("normalized", "ln1", global_state.active_block);

		activeComponent.name = "ln1";
	}

	const LN2Callback = async () => {
		if (input.isChanged === true) {
			await runModel(input.text);
		}

		await getProbDensity("resid_mid", null, global_state.active_block);
		await getProbDensity("normalized", "ln2", global_state.active_block);

		activeComponent.name = "ln2";
	}
</script>

<DottedBlockBase label="Transformer Blocks" inStyle="flex-col p-4">
	<ThemeNumberOptions
		count={12}
		bind:activeIndex={global_state.active_block}
		clickEventCb={null}
	/>
	<DottedBlockBase
		label="Block: {global_state.active_block}"
		borderSize={'1px'}
		titStyle="text-ti top-[-1.4rem]"
		inStyle="w-[30rem] h-[20rem] flex-row justify-between"
	>
		<div class="flex h-full flex-col items-start justify-evenly">
			<ElementBlockBase
				blockEle={AttentionHeads}
				blockStyle="p-2 min-w-[12rem] min-h-[10rem]"
				href={_transformerBlock[0].href}
				clickEventCb={getAttnPattern}
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
