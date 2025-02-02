<script lang="ts">
	import { fade } from 'svelte/transition';

	const { clickEvent = null, children = null } = $props();
</script>

<div
	transition:fade={{ duration: 200 }}
	class="fixed left-0 top-0 z-10 flex h-screen w-screen flex-col items-center justify-center bg-[#0004] backdrop-blur-[0.8px]"
	onclick={() => (clickEvent ? clickEvent() : () => {})}
>
	{#if children}
		{@render children()}
	{/if}
</div>
