<script lang="ts">
	import { ExpandOutline, MinimizeOutline } from 'flowbite-svelte-icons';
	import { fade } from 'svelte/transition';
	import QuickLink from './QuickLink.svelte';
	import DottedBlockBase from './DottedBlockBase.svelte';

	let {
		label = '',
		href = '/',
		expanded = $bindable<boolean>(),
		borderSize = '1.5px',
		inStyle = '',
		titStyle = '',
		children = null
	} = $props();

</script>

<DottedBlockBase {label} {borderSize} {inStyle} {titStyle}>
	<div class="absolute end-1 top-[0.2rem] flex flex-row items-center justify-evenly text-theme">
		<QuickLink {href} ostyle="!relative !text-theme" />

		<button onclick={() => (expanded = !expanded)}>
			{#if expanded}
				<MinimizeOutline size={'md'} />
			{:else}
				<ExpandOutline size={'md'} />
			{/if}
		</button>
	</div>
	{#if children}
		{@render children()}
	{/if}
</DottedBlockBase>

