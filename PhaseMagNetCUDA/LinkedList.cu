

#include "LinkedList.cuh"

template <typename T>
LinkedListNode<T>::LinkedListNode(T elem, LinkedListNode<T>* prev = nullptr, LinkedListNode<T>* next = nullptr) :
	elem(elem),
	prev(prev),
	next(next) {}

template <typename T>
LinkedListNode<T>::~LinkedListNode() {
	// recursively delete neighbors
	delete next; // nullptr is nop
	next = nullptr;
	delete prev;
	prev = nullptr;
} 

template <typename T>
const T LinkedListNode<T>::getElem(void) {
	return elem;
}

template <typename T>
void LinkedListNode<T>::setNext(LinkedListNode<T>* ptr) {
	next = ptr;
}

template <typename T>
void LinkedListNode<T>::setPrev(LinkedListNode<T>* ptr) {
	prev = ptr;
}

template <typename T>
LinkedListNode<T>* LinkedListNode<T>::getNext(void) {
	return next;
}

template <typename T>
LinkedListNode<T>* LinkedListNode<T>::getPrev(void) {
	return prev;
}

template <typename T>
bool LinkedListNode<T>::hasNext(void) {
	return next != nullptr;
}

template <typename T>
bool LinkedListNode<T>::hasPrev(void) {
	return prev != nullptr;
}

template <typename T>
LinkedList<T>::LinkedList() :
	head(nullptr),
	tail(nullptr),
	size(0) {}

template <typename T>
LinkedList<T>::~LinkedList() {
	delete head; // recursively delete neighbors
	head = nullptr;
	tail = nullptr;
}

template <typename T>
void LinkedList<T>::append(T elem) {
	// edge condition: head / tail are nullptr, size 0
	if (size == 0) {
		head = new LinkedListNode<T>(elem);
		tail = head;
	}
	else {
		tail->setNext(new LinkedListNode<T>(elem, tail));
		tail = tail->getNext();
	}
	++size;
}

template <typename T>
LinkedListNode<T>* LinkedList<T>::getHead(void) {
	return head;
}

template <typename T>
LinkedListNode<T>* LinkedList<T>::getTail(void) {
	return tail;
}

template <typename T>
bool LinkedList<T>::isEmpty(void) {
	return size == 0;
}