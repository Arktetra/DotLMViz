<script lang="ts">
	import { QuestionCircleSolid } from 'flowbite-svelte-icons';
	import Popup from './Popup.svelte';

	const {
		href = '/',
		blockStyle = '',
		clickEventCb = null,
        blockEle = null,
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
	class="relative m-2 flex cursor-pointer flex-col items-center justify-center rounded-md bg-theme p-2 text-theme-w transition-all duration-200 hover:bg-theme-alt {blockStyle}"
>
	<a {href} on:click|stopPropagation class="absolute end-1 top-1 text-theme-w">
		<QuestionCircleSolid size={'sm'} />
	</a>
    {#if children}
		{@render children()}
    {/if}
</div>

{#if blockEle && overlayState}
	<Popup onCloseCb={() => blockTrigger(false)} style="bg-theme-w justify-between p-0">
		{@render blockEle()}
	</Popup>
{/if}
