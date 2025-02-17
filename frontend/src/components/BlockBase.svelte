<script lang="ts">
	import Popup from './Popup.svelte';
	import { fade } from 'svelte/transition';
	import QuickLink from './QuickLink.svelte';
	import { global_state } from '../state.svelte';

	const {
		label = '',
		href = '/',
		width = '12rem',
		height = '5rem',
		style = '',
		clickEventCb = null,
		blockContent = null,
		children = null
	} = $props();

	let overlayState: boolean = $state(false);

	const blockTrigger = (newstate: boolean = false) => {
		global_state.ouputBlockState = true;
		overlayState = newstate;
		if (clickEventCb) clickEventCb();
	};
</script>

<div
	transition:fade={{ duration: 300 }}
	on:click={() => blockTrigger(true)}
	style="width:{width};height:{height};"
	class="relative m-2 flex cursor-pointer items-center justify-center rounded-md bg-theme/95 p-2 text-theme-w transition-all duration-200 hover:bg-theme-alt hover:tracking-wide {style}"
>
	<QuickLink {href} />
	{label}
	{#if children}
		{@render children()}
	{/if}
</div>

{#if blockContent && overlayState}
	<Popup onCloseCb={() => blockTrigger(false)}>
		{@render blockContent()}
	</Popup>
{/if}
