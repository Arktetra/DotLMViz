<script lang="ts">
	import { QuestionCircleSolid } from 'flowbite-svelte-icons';
	import Popup from './Popup.svelte';

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
	on:click={() => blockTrigger(true)}
	style="width:{width};height:{height};"
	class="relative m-2 flex cursor-pointer flex-col items-center justify-center rounded-md bg-theme p-2 text-theme-w transition-all duration-200 hover:scale-[102%] hover:bg-theme-alt {style}"
>
	<a {href} title={label} on:click|stopPropagation class="absolute end-1 top-1 text-theme-w">
		<QuestionCircleSolid size={'sm'} />
	</a>

	{label}
</div>

{#if children && overlayState}
	<Popup onCloseCb={() => blockTrigger(false)}>
		{@render children()}
	</Popup>
{/if}
