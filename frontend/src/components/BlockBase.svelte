<script lang="ts">
	import Popup from './Popup.svelte';
	import { fade } from 'svelte/transition';
	import QuickLink from './QuickLink.svelte';

	const {
		label = 'Untitled',
		href = '/',
		width = '12rem',
		height = '5rem',
		style = '',
		clickEventCb = null,
		children = null
	} = $props();

	let overlayState: boolean = $state(false);

	const blockTrigger = (newstate: boolean = false) => {
		overlayState = newstate;
		if (clickEventCb) clickEventCb();
	};
</script>

<div
	transition:fade={{ duration: 300 }}
	on:click={() => blockTrigger(true)}
	style="width:{width};height:{height};"
	class="relative m-2 flex cursor-pointer flex-col items-center justify-center rounded-md bg-theme p-2 text-theme-w transition-all duration-200 hover:scale-[102%] hover:bg-theme-alt {style}"
>
	<QuickLink {href} />
	{label}
</div>

{#if children && overlayState}
	<Popup onCloseCb={() => blockTrigger(false)}>
		{@render children()}
	</Popup>
{/if}
