<script lang="ts">
	import { scale } from 'svelte/transition';

	const { label = '', children = null, clickEventCb = null } = $props();

	const btnClick = () => {
		if (clickEventCb) clickEventCb();
	};
</script>

<button
	onclick={btnClick}
	transition:scale={{ duration: 500 }}
	class="m-2 min-h-[2rem] min-w-[2rem] rounded-md bg-theme p-2 px-4 text-theme-w transition-transform duration-300 hover:scale-[103%] hover:shadow-md active:bg-theme-alt"
>
	{label}
	{#if children}
		{@render children()}
	{/if}
</button>
