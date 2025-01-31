<script>
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import BlockBase from '../components/BlockBase.svelte';
	import { embedCallback, posEmbedCallback } from '../callbacks.svelte';

	const _embeddings = [
		{
			label: 'Token Embedding',
			href: '/read/token-embedding',
			callback: embedCallback
		},
		{
			label: 'Positional Embedding',
			href: '/read/positional-embedding',
			callback: posEmbedCallback
		}
	];
</script>

<DottedBlockBase label="Embeddings" inStyle="h-full flex-col justify-center items-center">
	{#each _embeddings as item}
		<BlockBase label={item.label} href={item.href} clickEventCb={item.callback}>
			<h1 class="bg-theme p-4">{item.label}</h1>
		</BlockBase>
	{/each}
</DottedBlockBase>
