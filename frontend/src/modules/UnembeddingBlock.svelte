<script lang="ts">
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import BlockBase from '../components/BlockBase.svelte';
	import { global_state } from '../state.svelte';
	import ElementBlockBase from '../components/ElementBlockBase.svelte';
</script>

<DottedBlockBase label="Unembedding" inStyle="flex-col justify-evenly items-center h-full">
	{#if global_state.active_block == 11}
	<ElementBlockBase blockStyle="w-[12rem] p-2 min-w-[4rem] min-h-[4rem]" href={'/read/layernorm'}>
		<span>LN</span>
	</ElementBlockBase>
	{/if}
	<BlockBase label="Unembedding" width={'12rem'} height={global_state.active_block == 11 ? '18rem' : '25rem'} href={'/read/unembedding'}
	></BlockBase>
</DottedBlockBase>
