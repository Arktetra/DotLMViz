<script>
	import DottedBlockBase from '../components/DottedBlockBase.svelte';
	import BlockBase from '../components/BlockBase.svelte';
	import { getAct } from '../routes/fetch.svelte';
	import { activeComponent } from '../state.svelte';

	const getEmbedOutput = async () => {
		await getAct("embed", null, null);
		activeComponent.name = "embed";
	}

	const getPosEmbedOutput = async () => {
		await getAct("pos_embed", null, null);
		activeComponent.name = "pos_embed";
	}

	const _embeddings = [
		{
			label: 'Token Embedding',
			href: '/read/token-embedding',
			callback: getEmbedOutput
		},
		{
			label: 'Positional Embedding',
			href: '/read/positional-embedding',
			callback: getPosEmbedOutput
		}
	];


</script>

<DottedBlockBase label="Embeddings" inStyle="flex-col">
	{#each _embeddings as item}
		<BlockBase label={item.label} href={item.href} clickEventCb={item.callback}>
			<h1 class="bg-theme p-4">{item.label}</h1>
		</BlockBase>
	{/each}
</DottedBlockBase>
