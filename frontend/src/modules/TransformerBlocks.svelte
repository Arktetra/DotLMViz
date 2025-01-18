<script lang="ts">
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import BlockBase from '../components/BlockBase.svelte';
	import ThemeNumberOptions from '../components/ThemeNumberOptions.svelte';

	const _transformerBlock = [
		{
			label: 'Attention Head',
			href: '/readings/attentionhead'
		},
		{
			label: 'MLP',
			href: '/readings/mlp'
		}
	];

	let _activeBlock = $state(0);
</script>

<DottedBlockBase label="Transformer Blocks" inStyle="flex-col p-4">
	<span class="text-md my-2 uppercase text-theme">Blocks</span>
	<ThemeNumberOptions count={12} bind:activeIndex={_activeBlock} />
	<DottedBlockBase
		label="Block: {_activeBlock}"
		borderSize={'1px'}
		titStyle="text-ti top-[-1.4rem]"
		inStyle="w-[30rem] h-[20rem] flex-row justify-between"
	>
		{#each _transformerBlock as item}
			<div class="flex h-full flex-col items-start justify-evenly">
				<BlockBase label={item.label} height={'10rem'} href={item.href}>
					<h1 class="bg-theme p-4">{item.label}</h1>
				</BlockBase>
				<BlockBase label="LN" width={'4rem'} height={'4rem'} href={'/readings/layernorm'}>
					<h1 class="bg-theme p-4">LayerNorm</h1>
				</BlockBase>
			</div>
		{/each}
	</DottedBlockBase>
</DottedBlockBase>
