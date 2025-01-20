<script lang="ts">
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import BlockBase from '../components/BlockBase.svelte';
	import ThemeNumberOptions from '../components/ThemeNumberOptions.svelte';
	import ElementBlockBase from '../components/ElementBlockBase.svelte';
	import AttentionHeads from './AttentionHeads.svelte';
	import Mlp from './MLP.svelte';

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

	let _activeBlock = $state(0);
</script>

<DottedBlockBase label="Transformer Blocks" inStyle="flex-col p-4">
	<ThemeNumberOptions count={12} bind:activeIndex={_activeBlock} />
	<DottedBlockBase
		label="Block: {_activeBlock}"
		borderSize={'1px'}
		titStyle="text-ti top-[-1.4rem]"
		inStyle="w-[30rem] h-[20rem] flex-row justify-between"
	>
		<div class="flex h-full flex-col items-start justify-evenly">
			<ElementBlockBase blockEle={AttentionHeads} blockStyle="p-2 min-w-[12rem] min-h-[10rem]" href={_transformerBlock[0].href}>
				<span>{_transformerBlock[0].label}</span>
			</ElementBlockBase>
			<ElementBlockBase blockStyle="p-2 min-w-[4rem] min-h-[4rem]" href={'/read/layernorm'}>
				<span>LN</span>
			</ElementBlockBase>
		</div>
		<div class="flex h-full flex-col items-start justify-evenly">
			<ElementBlockBase blockEle={Mlp} blockStyle="p-4 min-w-[12rem] min-h-[10rem]" href={_transformerBlock[1].href}>
				<span>{_transformerBlock[1].label}</span>
			</ElementBlockBase>
			<ElementBlockBase blockStyle="p-2 min-w-[4rem] min-h-[4rem]" href={'/read/layernorm'}>
				<span>LN</span>
			</ElementBlockBase>
		</div>
	</DottedBlockBase>

</DottedBlockBase>
